//! Register as a child of `runtime.rs`, using the physical packed-onehot path.
//! K16 opening arithmetic stays on Metal; nonce selection stays with the prover.

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::packed_onehot::PackedOneHotCommitView;
use akita_error::checked;
use akita_prover::compute::DecomposeFoldPlan;

const SOURCE: &str = include_str!("k16.metal");
const KERNEL: &str = "akita_k16_decompose_fold";
const THREADS: usize = 256;
const MAX_LANE_STAGING_BYTES: usize = 16 << 20;
const MAX_POSITION_TILE: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct K16FoldParams {
    num_columns: u64,
    num_positions: u64,
    position_start: u64,
    position_count: u64,
    blocks_per_column: u64,
    ring_dimension: u64,
    rows_per_ring: u64,
    zero_column_mask: u64,
    tiled_source: u64,
}
const _: [(); 72] = [(); size_of::<K16FoldParams>()];

struct Challenges {
    offsets: Vec<u32>,
    positions: Vec<u16>,
    coefficients: Vec<i8>,
    blocks_per_column: usize,
}

fn validate<const D: usize>(
    source: PackedOneHotCommitView<'_>,
    plan: DecomposeFoldPlan<'_>,
) -> Result<Challenges, MetalCommitError> {
    if !matches!(D, 64 | 128 | 256 | 512)
        || source.onehot_k() != 16
        || source.column_capacity() != 64
        || plan.num_positions_per_block == 0
        || plan.num_digits == 0
    {
        return Err(MetalCommitError::UnsupportedShape(
            "K16 fold requires capacity64, D64/128/256/512 and nonzero positions/digits".into(),
        ));
    }
    let rows_per_block = checked::product([plan.num_positions_per_block, D / 16])
        .ok_or(MetalCommitError::ShapeOverflow("K16 rows per block"))?;
    let blocks_per_column = checked::exact_div(source.num_rows(), rows_per_block)
        .filter(|blocks| *blocks != 0)
        .ok_or_else(|| {
            MetalCommitError::UnsupportedShape("K16 trace blocks are not integral".into())
        })?;
    let expected = checked::product([64, blocks_per_column])
        .ok_or(MetalCommitError::ShapeOverflow("K16 challenge count"))?;
    if plan.challenges.len() != expected {
        return Err(MetalCommitError::UnsupportedShape(
            "K16 challenge count disagrees with the packed capacity".into(),
        ));
    }
    let live = checked::product([source.num_columns(), blocks_per_column])
        .ok_or(MetalCommitError::ShapeOverflow("K16 live challenges"))?;
    let mut absolute_sum = 0usize;
    let mut terms = 0usize;
    for (index, challenge) in plan.challenges.iter().enumerate() {
        challenge
            .validate::<D>()
            .map_err(|error| MetalCommitError::UnsupportedShape(error.to_string()))?;
        if index < live {
            terms = terms
                .checked_add(challenge.positions.len())
                .ok_or(MetalCommitError::ShapeOverflow("K16 challenge terms"))?;
            let absolute = checked::sum(
                challenge
                    .coeffs
                    .iter()
                    .map(|value| i32::from(*value).unsigned_abs() as usize),
            )
            .ok_or(MetalCommitError::ShapeOverflow(
                "K16 challenge absolute sum",
            ))?;
            absolute_sum = absolute_sum
                .checked_add(absolute)
                .ok_or(MetalCommitError::ShapeOverflow("K16 accumulation bound"))?;
        }
    }
    // Each ring position has D/K selector rows for each live challenge. This
    // bounds every partial signed atomic sum, regardless of update ordering.
    let bound = checked::product([D / 16, absolute_sum]).ok_or(MetalCommitError::ShapeOverflow(
        "K16 signed accumulation bound",
    ))?;
    if bound > i32::MAX as usize || terms > u32::MAX as usize {
        return Err(MetalCommitError::UnsupportedShape(
            "K16 signed accumulation or challenge indexing exceeds i32/u32".into(),
        ));
    }
    let offset_count =
        checked::sum([live, 1]).ok_or(MetalCommitError::ShapeOverflow("K16 challenge offsets"))?;
    let mut offsets = Vec::with_capacity(offset_count);
    let mut positions = Vec::with_capacity(terms);
    let mut coefficients = Vec::with_capacity(terms);
    offsets.push(0);
    for challenge in &plan.challenges[..live] {
        // SparseChallenge validation above enforces paired lengths, distinct
        // positions, and positions below D. Variable weights need no expansion.
        positions.extend(challenge.positions.iter().map(|position| *position as u16));
        coefficients.extend_from_slice(&challenge.coeffs);
        offsets.push(positions.len() as u32);
    }
    Ok(Challenges {
        offsets,
        positions,
        coefficients,
        blocks_per_column,
    })
}

fn pipeline(runtime: &MetalRuntime) -> Result<ComputePipelineState, MetalCommitError> {
    // Cache only an owned pipeline by physical device identity. There is no
    // source-address cache and no borrowed input retained between attempts.
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipelineState>>> = OnceLock::new();
    let mut pipelines = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| MetalCommitError::PoisonedLock)?;
    let device_id = runtime.device.registry_id();
    if let Some(pipeline) = pipelines.get(&device_id) {
        return Ok(pipeline.clone());
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = runtime
        .device
        .new_library_with_source(SOURCE, &options)
        .map_err(MetalCommitError::LibraryCompilation)?;
    let function =
        library
            .get_function(KERNEL, None)
            .map_err(|message| MetalCommitError::FunctionLookup {
                name: KERNEL,
                message,
            })?;
    let pipeline = runtime
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalCommitError::PipelineCompilation {
            name: KERNEL,
            message,
        })?;
    if pipeline.max_total_threads_per_threadgroup() < THREADS as u64
        || pipeline.thread_execution_width() != 32
        || pipeline.static_threadgroup_memory_length()
            > runtime.device.max_threadgroup_memory_length()
    {
        return Err(MetalCommitError::UnsupportedShape(
            "K16 fold pipeline resources are unavailable".into(),
        ));
    }
    pipelines.insert(device_id, pipeline.clone());
    Ok(pipeline)
}

impl MetalRuntime {
    pub(crate) fn dispatch_packed_k16_decompose_fold<const D: usize>(
        &self,
        source: PackedOneHotCommitView<'_>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        let challenges = validate::<D>(source, plan)?;
        let positions = plan.num_positions_per_block;
        let rows_per_ring = D / 16;
        let source_bytes_per_position = checked::product([
            challenges.blocks_per_column,
            rows_per_ring,
            source.num_columns(),
        ])
        .ok_or(MetalCommitError::ShapeOverflow(
            "K16 position staging bytes",
        ))?;
        let zero_copy = source
            .lanes()
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && source
                .lanes()
                .len()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let tile_positions = if zero_copy {
            positions.min(MAX_POSITION_TILE)
        } else {
            (MAX_LANE_STAGING_BYTES / source_bytes_per_position)
                .min(positions)
                .min(MAX_POSITION_TILE)
        };
        if tile_positions == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "one K16 position exceeds the bounded lane staging window".into(),
            ));
        }
        let output_count = checked::product([positions, D])
            .ok_or(MetalCommitError::ShapeOverflow("K16 fold output"))?;
        let expanded_count = checked::product([output_count, plan.num_digits])
            .ok_or(MetalCommitError::ShapeOverflow("K16 expanded witness"))?;
        // The fixed staging cap is not a process-memory guarantee. The caller
        // retains the proof guard and admits the complete host witness too.
        let output_bytes = checked::product([tile_positions, D, size_of::<i32>()])
            .ok_or(MetalCommitError::ShapeOverflow("K16 tile output bytes"))?;
        let lane_bytes = if zero_copy {
            0
        } else {
            checked::product([tile_positions, source_bytes_per_position])
                .ok_or(MetalCommitError::ShapeOverflow("K16 lane staging bytes"))?
        };
        let metadata_bytes = checked::sum([
            std::mem::size_of_val(challenges.offsets.as_slice()),
            std::mem::size_of_val(challenges.positions.as_slice()).max(size_of::<u16>()),
            std::mem::size_of_val(challenges.coefficients.as_slice()).max(size_of::<i8>()),
            std::mem::size_of_val(source.active_zero_rows()),
            usize::from(source.active_zero_rows().is_empty()) * size_of::<u64>(),
        ])
        .ok_or(MetalCommitError::ShapeOverflow("K16 metadata bytes"))?;
        let allocation_bytes = checked::sum([output_bytes, lane_bytes, metadata_bytes])
            .ok_or(MetalCommitError::ShapeOverflow("K16 device workspace"))?;
        if allocation_bytes as u64 > self.device.recommended_max_working_set_size() {
            return Err(MetalCommitError::BufferTooLong {
                requested: allocation_bytes as u64,
                maximum: self.device.recommended_max_working_set_size(),
            });
        }
        let kernel = pipeline(self)?;
        let buffer_start = Instant::now();
        let borrowed_lanes = if zero_copy {
            Some(self.packed_lane_buffer(source.lanes())?)
        } else {
            None
        };
        let staged_lanes = if zero_copy {
            None
        } else {
            Some(self.shared_buffer(lane_bytes)?)
        };
        let source_buffer = borrowed_lanes
            .as_ref()
            .map(|lanes| &lanes.buffer)
            .or(staged_lanes.as_ref())
            .ok_or_else(|| {
                MetalCommitError::UnsupportedShape("K16 source buffer is missing".into())
            })?;
        let dummy_mask = [0u64];
        let masks = if source.active_zero_rows().is_empty() {
            &dummy_mask[..]
        } else {
            source.active_zero_rows()
        };
        let mask_buffer = self.shared_buffer_from_slice(masks)?;
        let offset_buffer = self.shared_buffer_from_slice(&challenges.offsets)?;
        // Empty challenges are still a valid zero sum if accepted by the
        // canonical SparseChallenge validator. Supply initialized dummy slots.
        let dummy_position = [0u16];
        let dummy_coefficient = [0i8];
        let position_buffer =
            self.shared_buffer_from_slice(if challenges.positions.is_empty() {
                &dummy_position[..]
            } else {
                &challenges.positions
            })?;
        let coefficient_buffer =
            self.shared_buffer_from_slice(if challenges.coefficients.is_empty() {
                &dummy_coefficient[..]
            } else {
                &challenges.coefficients
            })?;
        let output = self.shared_buffer(output_bytes)?;
        let buffer_setup = buffer_start.elapsed();
        let mut centered_coefficients = vec![0i32; output_count];
        let mut command_wall = Duration::ZERO;
        let mut gpu = Duration::ZERO;
        let mut all_gpu_timestamps = true;
        let mut readback_copy = Duration::ZERO;
        let mut staging_time = Duration::ZERO;
        let mut command_buffers = 0usize;
        for position_start in (0..positions).step_by(tile_positions) {
            let count = tile_positions.min(positions - position_start);
            if !zero_copy {
                let start = Instant::now();
                let block_tile_bytes =
                    checked::product([count, rows_per_ring, source.num_columns()])
                        .ok_or(MetalCommitError::ShapeOverflow("K16 block tile bytes"))?;
                for block in 0..challenges.blocks_per_column {
                    let ring_start = checked::mul_add(block, positions, position_start)
                        .ok_or(MetalCommitError::ShapeOverflow("K16 source ring offset"))?;
                    let source_start =
                        checked::product([ring_start, rows_per_ring, source.num_columns()])
                            .ok_or(MetalCommitError::ShapeOverflow("K16 source byte offset"))?;
                    let source_range = checked::range(source_start, block_tile_bytes)
                        .ok_or(MetalCommitError::ShapeOverflow("K16 source byte range"))?;
                    let bytes = source.lanes().get(source_range).ok_or_else(|| {
                        MetalCommitError::UnsupportedShape(
                            "K16 source tile exceeds borrowed lanes".into(),
                        )
                    })?;
                    let destination = checked::product([block, block_tile_bytes])
                        .ok_or(MetalCommitError::ShapeOverflow("K16 staging destination"))?;
                    // SAFETY: validated blocks partition the immutable source;
                    // staging owns >= B*count*R*C bytes. The prior command has
                    // completed before this shared buffer is overwritten.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            source_buffer.contents().cast::<u8>().add(destination),
                            bytes.len(),
                        );
                    }
                }
                staging_time += start.elapsed();
            }
            let params = K16FoldParams {
                num_columns: source.num_columns() as u64,
                num_positions: positions as u64,
                position_start: position_start as u64,
                position_count: count as u64,
                blocks_per_column: challenges.blocks_per_column as u64,
                ring_dimension: D as u64,
                rows_per_ring: rows_per_ring as u64,
                zero_column_mask: source.zero_column_mask(),
                tiled_source: u64::from(!zero_copy),
            };
            let (wall, gpu_time) = autoreleasepool(|| {
                let command = self.queue.new_command_buffer();
                command.set_label("Akita K16 packed decompose-fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&kernel);
                for (index, buffer) in [
                    source_buffer,
                    &mask_buffer,
                    &offset_buffer,
                    &position_buffer,
                    &coefficient_buffer,
                    &output,
                ]
                .into_iter()
                .enumerate()
                {
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
                set_inline_bytes(encoder, 6, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(count as u64, 1, 1),
                    MTLSize::new(THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                // Canonical helper records submission, waits and checks status.
                // All source borrows and buffers outlive this completion.
                complete_command(command)
            })?;
            command_wall += wall;
            if let Some(time) = gpu_time {
                gpu += time;
            } else {
                all_gpu_timestamps = false;
            }
            command_buffers += 1;
            let start = Instant::now();
            // SAFETY: completed command initialized count*D i32s in the
            // bounded output buffer. Host ranges are disjoint and in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output.contents().cast::<i32>(),
                    centered_coefficients.as_mut_ptr().add(position_start * D),
                    count * D,
                );
            }
            readback_copy += start.elapsed();
        }
        tracing::debug!(
            ring_dimension = D,
            rows = source.num_rows(),
            columns = source.num_columns(),
            blocks_per_column = challenges.blocks_per_column,
            tile_positions,
            command_buffers,
            lane_staging_bytes = lane_bytes,
            allocation_bytes,
            source_buffer_bytes = source_buffer.length(),
            source_bytes = source.lanes().len(),
            host_output_bytes = std::mem::size_of_val(centered_coefficients.as_slice()),
            expanded_host_coefficients = expanded_count,
            input_zero_copy = zero_copy,
            gpu_timestamps_complete = all_gpu_timestamps,
            gpu_seconds = gpu.as_secs_f64(),
            "completed K16 packed Metal fold"
        );
        Ok(PackedDecomposeFoldDispatchOutcome {
            centered_coefficients,
            timings: DispatchTimings {
                buffer_setup: buffer_setup + staging_time,
                command_wall,
                gpu: all_gpu_timestamps.then_some(gpu),
                readback_copy,
            },
            consumer_time: Duration::ZERO,
            allocation_bytes,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "small checked K16 Metal parity fixtures"
)]
mod tests {
    use super::*;
    use akita_prover::backend::poly_helpers::build_decompose_fold_witness;
    use akita_prover::compute::OpeningFoldKernel;
    use akita_prover::{CpuBackend, OneHotPoly, RootOpeningSource};

    struct AlignedLanes {
        pointer: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
    }
    impl AlignedLanes {
        fn new(bytes: &[u8]) -> Self {
            let layout =
                std::alloc::Layout::from_size_align(bytes.len(), PACKED_ONEHOT_BUFFER_ALIGNMENT)
                    .unwrap();
            // SAFETY: nonzero checked layout; this test owner releases the
            // allocation only after every borrowed GPU command completes.
            let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr(), bytes.len());
            }
            Self { pointer, layout }
        }
        fn bytes(&self) -> &[u8] {
            // SAFETY: allocation is initialized and owned for this borrow.
            unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
        }
    }
    impl Drop for AlignedLanes {
        fn drop(&mut self) {
            // SAFETY: exact original allocation/layout, never cached by dispatch.
            unsafe {
                std::alloc::dealloc(self.pointer.as_ptr(), self.layout);
            }
        }
    }

    fn parity<const D: usize>(rows: usize, positions: usize, aligned: bool) {
        const COLUMNS: usize = 5;
        let runtime = MetalRuntime::new().unwrap();
        let bytes = (0..rows * COLUMNS)
            .map(|index| {
                let row = index / COLUMNS;
                if row >= rows - rows / 4 || index % 7 == 0 {
                    0
                } else {
                    (index % 15 + 1) as u8
                }
            })
            .collect::<Vec<_>>();
        let owner = aligned.then(|| AlignedLanes::new(&bytes));
        let lanes = owner.as_ref().map_or(bytes.as_slice(), AlignedLanes::bytes);
        let active = (0..rows.div_ceil(64))
            .map(|word| {
                (0..64).fold(0u64, |bits, bit| {
                    let row = word * 64 + bit;
                    if row < rows - rows / 4 && row % 3 == 0 {
                        bits | (1u64 << bit)
                    } else {
                        bits
                    }
                })
            })
            .collect::<Vec<_>>();
        let source = PackedOneHotCommitView::new_with_active_zero_rows(
            16, 64, COLUMNS, lanes, &active, 0b10101,
        )
        .unwrap();
        // Materialize only this tiny fixture, including all inactive capacity
        // columns. The production dispatcher never expands one-hot indices.
        let indices = (0..64)
            .flat_map(|column| {
                (0..rows).map({
                    let active = &active;
                    move |row| {
                        if column >= COLUMNS {
                            return None;
                        }
                        let hot = lanes[row * COLUMNS + column];
                        let committed_zero = (0b10101u64 >> column) & 1 != 0
                            && (active[row / 64] >> (row % 64)) & 1 != 0;
                        (hot != 0 || committed_zero).then_some(hot)
                    }
                })
            })
            .collect();
        let reference = OneHotPoly::<F, u8>::new(16, indices).unwrap();
        let blocks = rows / ((D / 16) * positions);
        for attempt in 0..2 {
            let challenges = (0..64 * blocks)
                .map(|block| {
                    // Distinct positions include a signed wrap at D-1 and vary
                    // weights among challenges without dense rotation tables.
                    let positions = if (block + attempt) % 2 == 0 {
                        vec![0, (D - 1) as u32]
                    } else {
                        vec![1, (D / 2) as u32, (D - 1) as u32]
                    };
                    let coefficients = if positions.len() == 2 {
                        vec![1, -2]
                    } else {
                        vec![-1, 2, 1]
                    };
                    akita_challenges::SparseChallenge {
                        positions: positions.into(),
                        coeffs: coefficients.into(),
                    }
                })
                .collect::<Vec<_>>();
            let plan = DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: positions,
                num_digits: 2,
                log_basis: 3,
            };
            let expected = CpuBackend::DEFAULT
                .decompose_fold(
                    None,
                    <OneHotPoly<F, u8> as RootOpeningSource<F, D>>::opening_view(&reference)
                        .unwrap(),
                    plan,
                )
                .unwrap();
            let actual = runtime
                .dispatch_packed_k16_decompose_fold::<D>(source, plan)
                .unwrap();
            for (actual, expected) in actual
                .centered_coefficients
                .as_chunks::<D>()
                .0
                .iter()
                .zip(expected.centered_coeffs_trusted::<D>().iter().step_by(2))
            {
                assert_eq!(actual, expected);
            }
            let centered = actual
                .centered_coefficients
                .as_chunks::<D>()
                .0
                .iter()
                .flat_map(|coefficients| {
                    let mut row = [0i32; D];
                    row.copy_from_slice(coefficients);
                    [row, [0i32; D]]
                })
                .collect();
            let modulus = (-F::one()).to_canonical_u128() + 1;
            let witness = build_decompose_fold_witness::<F, D>(centered, modulus);
            assert_eq!(witness, expected);
            assert!(actual.timings.gpu.is_some());
            assert!(actual.allocation_bytes < MAX_LANE_STAGING_BYTES + (4 << 20));
        }
        // Source and challenge borrows remain alive across repeated attempts;
        // the dispatcher completed every command before either can be dropped.
    }

    #[test]
    fn k16_d64_gpu_matches_cpu_witness_across_attempts() {
        parity::<64>(256, 8, false);
    }
    #[test]
    fn k16_d128_gpu_matches_cpu_witness_across_attempts() {
        parity::<128>(256, 8, false);
    }
    #[test]
    fn k16_d256_gpu_matches_cpu_witness_across_attempts() {
        parity::<256>(256, 8, false);
    }
    #[test]
    fn k16_d512_gpu_matches_cpu_witness_across_attempts() {
        parity::<512>(256, 8, false);
    }
    #[test]
    fn k16_position_tiles_cover_seams_and_padded_rows() {
        parity::<64>(8192, 2048, false);
    }
    #[test]
    fn k16_aligned_borrow_matches_cpu_and_releases_after_attempts() {
        parity::<512>(32768, 1024, true);
    }

    #[test]
    fn k16_validation_rejects_geometry_and_challenge_mismatches() {
        let lanes = vec![1u8; 128];
        let source = PackedOneHotCommitView::new(16, 64, 1, &lanes).unwrap();
        let challenge = akita_challenges::SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        let challenges = vec![challenge; 64];
        let plan = DecomposeFoldPlan {
            challenges: &challenges,
            num_positions_per_block: 32,
            num_digits: 1,
            log_basis: 3,
        };
        assert!(validate::<32>(source, plan).is_err());
        assert!(validate::<512>(source, plan).is_err());
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                num_digits: 0,
                ..plan
            }
        )
        .is_err());
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                num_positions_per_block: usize::MAX,
                ..plan
            }
        )
        .is_err());
        let mut malformed = challenges.clone();
        malformed[0] = akita_challenges::SparseChallenge {
            positions: vec![64].into(),
            coeffs: vec![1].into(),
        };
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                challenges: &malformed,
                ..plan
            }
        )
        .is_err());
        malformed[0] = akita_challenges::SparseChallenge {
            positions: vec![1, 1].into(),
            coeffs: vec![1, -1].into(),
        };
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                challenges: &malformed,
                ..plan
            }
        )
        .is_err());
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                num_positions_per_block: 31,
                ..plan
            }
        )
        .is_err());
        assert!(validate::<64>(
            source,
            DecomposeFoldPlan {
                challenges: &challenges[..63],
                ..plan
            }
        )
        .is_err());
    }
}
