use super::*;

const THREADS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct WitnessPackParams {
    ring_d: u64,
    tile_width: u64,
    fold_planes: u64,
    log_basis: u64,
    bit_width: u64,
}

pub(crate) struct WitnessPackOutcome {
    pub(crate) bytes: Vec<u8>,
    pub(crate) digit_len: usize,
    pub(crate) negative_abs_max: u8,
    pub(crate) positive_max: u8,
    pub(crate) timings: DispatchTimings,
}

impl MetalRuntime {
    pub(crate) fn pack_centered_witness_z<const D: usize>(
        &self,
        centered: &[i32],
        fold_planes: usize,
        log_basis: u32,
        bit_width: u8,
    ) -> Result<WitnessPackOutcome, MetalCommitError> {
        if !matches!(D, 64 | 128 | 256 | 512)
            || !centered.len().is_multiple_of(D)
            || !(1..=8).contains(&log_basis)
            || !(1..=8).contains(&bit_width)
            || u32::from(bit_width) < log_basis
            || fold_planes > ((128 + log_basis) / log_basis) as usize
        {
            return Err(MetalCommitError::UnsupportedShape(
                "invalid centered witness Z packing geometry".into(),
            ));
        }
        let digit_len =
            centered
                .len()
                .checked_mul(fold_planes)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed witness Z digit extent",
                ))?;
        let encoded_bits = digit_len.checked_mul(usize::from(bit_width)).ok_or(
            MetalCommitError::ShapeOverflow("packed witness Z bit extent"),
        )?;
        let encoded_len = encoded_bits.div_ceil(8);
        let zero_timings = DispatchTimings {
            buffer_setup: Duration::ZERO,
            command_wall: Duration::ZERO,
            gpu: Some(Duration::ZERO),
            readback_copy: Duration::ZERO,
        };
        if digit_len == 0 {
            return Ok(WitnessPackOutcome {
                bytes: Vec::new(),
                digit_len,
                negative_abs_max: 0,
                positive_max: 0,
                timings: zero_timings,
            });
        }
        let tile_width = D.min(if fold_planes > 128 { 128 } else { THREADS });
        let tiles = centered.len() / tile_width;
        if tiles > u32::MAX as usize {
            return Err(MetalCommitError::ShapeOverflow(
                "packed witness Z tile extent",
            ));
        }
        let scratch_bytes = fold_planes * tile_width;
        if self
            .witness_pack_pipeline
            .max_total_threads_per_threadgroup()
            < THREADS as u64
            || self.witness_pack_pipeline.thread_execution_width() != 32
            || scratch_bytes as u64 > self.device.max_threadgroup_memory_length()
        {
            return Err(MetalCommitError::UnsupportedShape(
                "witness Z packing requires 256-thread groups and admitted shared scratch".into(),
            ));
        }
        self.validate_buffer_length(size_of_val(centered))?;
        self.validate_buffer_length(encoded_len)?;
        autoreleasepool(|| {
            let setup_start = Instant::now();
            let source = self.shared_buffer_from_slice(centered)?;
            let packed = self.shared_buffer(encoded_len)?;
            let bounds = self.shared_buffer_from_slice(&[0u32; 2])?;
            let params = WitnessPackParams {
                ring_d: D as u64,
                tile_width: tile_width as u64,
                fold_planes: fold_planes as u64,
                log_basis: u64::from(log_basis),
                bit_width: u64::from(bit_width),
            };
            let buffer_setup = setup_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita centered recursive witness Z packing");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.witness_pack_pipeline);
            encoder.set_buffer(0, Some(&source), 0);
            encoder.set_buffer(1, Some(&packed), 0);
            encoder.set_buffer(2, Some(&bounds), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.set_threadgroup_memory_length(0, scratch_bytes as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new(tiles as u64, 1, 1),
                MTLSize::new(THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: every output byte is written exactly once by the completed
            // dispatch; the two aligned bounds slots were initialized on upload.
            let (bytes, extrema) = unsafe {
                (
                    std::slice::from_raw_parts(packed.contents().cast::<u8>(), encoded_len)
                        .to_vec(),
                    *bounds.contents().cast::<[u32; 2]>(),
                )
            };
            let negative_abs_max = u8::try_from(extrema[0]).map_err(|_| {
                MetalCommitError::UnsupportedShape("witness Z negative bound exceeds i8".into())
            })?;
            let positive_max = u8::try_from(extrema[1]).map_err(|_| {
                MetalCommitError::UnsupportedShape("witness Z positive bound exceeds i8".into())
            })?;
            Ok(WitnessPackOutcome {
                bytes,
                digit_len,
                negative_abs_max,
                positive_max,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }
}
