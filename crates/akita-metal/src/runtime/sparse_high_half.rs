use super::*;
use akita_algebra::CyclotomicRing;

const THREADS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SparseHighHalfTerm {
    pub(crate) challenge_index: u64,
    pub(crate) shift: u32,
    pub(crate) coefficient: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseHighHalfRow {
    pub(crate) source_start: u64,
    pub(crate) source_end: u64,
    pub(crate) first_ring: u64,
    pub(crate) ring_stride: u64,
    pub(crate) live_challenges: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SparseHighHalfParams {
    ring_d: u64,
    batch_count: u64,
    term_count: u64,
}

pub(crate) struct SparseHighHalfOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

impl MetalRuntime {
    pub(crate) fn validate_sparse_high_half_buffers(
        &self,
        source_coefficients: usize,
        term_count: usize,
        row_count: usize,
        ring_d: usize,
    ) -> Result<(), MetalCommitError> {
        let output_coefficients =
            row_count
                .checked_mul(ring_d)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "sparse high-half output extent",
                ))?;
        if !matches!(ring_d, 64 | 128 | 256 | 512)
            || output_coefficients > u32::MAX as usize
            || self
                .sparse_high_half_pipeline
                .max_total_threads_per_threadgroup()
                < THREADS as u64
            || self.sparse_high_half_pipeline.thread_execution_width() != 32
            || (THREADS * size_of::<Fp128Limbs>()) as u64
                > self.device.max_threadgroup_memory_length()
        {
            return Err(MetalCommitError::UnsupportedShape(
                "sparse high-half batch geometry".into(),
            ));
        }
        for (count, width) in [
            (source_coefficients.max(1), size_of::<Fp128Limbs>()),
            (term_count.max(1), size_of::<SparseHighHalfTerm>()),
            (row_count.max(1), size_of::<SparseHighHalfRow>()),
            (output_coefficients.max(1), size_of::<Fp128Limbs>()),
        ] {
            let bytes = count
                .checked_mul(width)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "sparse high-half buffer size",
                ))?;
            self.validate_buffer_length(bytes)?;
        }
        Ok(())
    }

    pub(crate) fn sparse_high_half_batch<const D: usize>(
        &self,
        sources: &[&[CyclotomicRing<F, D>]],
        challenges: usize,
        terms: &[SparseHighHalfTerm],
        rows: &[SparseHighHalfRow],
        subtract: &[Fp128Limbs],
    ) -> Result<SparseHighHalfOutcome, MetalCommitError> {
        let source_coefficients = sources.iter().try_fold(0usize, |total, source| {
            source
                .len()
                .checked_mul(D)
                .and_then(|count| total.checked_add(count))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "sparse high-half source extent",
                ))
        })?;
        let output_coefficients =
            rows.len()
                .checked_mul(D)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "sparse high-half output extent",
                ))?;
        self.validate_sparse_high_half_buffers(source_coefficients, terms.len(), rows.len(), D)?;
        if subtract.len() != output_coefficients
            || terms.iter().any(|term| {
                term.challenge_index >= challenges as u64
                    || term.shift as usize >= D
                    || !(-128..=127).contains(&term.coefficient)
            })
        {
            return Err(MetalCommitError::UnsupportedShape(
                "sparse high-half term/subtraction extent".into(),
            ));
        }
        for row in rows {
            if row.live_challenges > challenges as u64
                || row.source_start > row.source_end
                || row.source_end > source_coefficients as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "sparse high-half live challenge extent".into(),
                ));
            }
            if row.live_challenges != 0 {
                let end = (row.live_challenges - 1)
                    .checked_mul(row.ring_stride)
                    .and_then(|offset| row.first_ring.checked_add(offset))
                    .and_then(|ring| ring.checked_add(1))
                    .and_then(|count| count.checked_mul(D as u64))
                    .and_then(|count| row.source_start.checked_add(count))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "sparse high-half strided source end",
                    ))?;
                if end > row.source_end {
                    return Err(MetalCommitError::UnsupportedShape(
                        "sparse high-half source lies outside upload".into(),
                    ));
                }
            }
        }
        if rows.is_empty() {
            return Ok(SparseHighHalfOutcome {
                coefficients: Vec::new(),
                allocation_bytes: 0,
                timings: DispatchTimings {
                    buffer_setup: Duration::ZERO,
                    command_wall: Duration::ZERO,
                    gpu: Some(Duration::ZERO),
                    readback_copy: Duration::ZERO,
                },
            });
        }
        // Every logical shape and device extent is admitted before allocating GPU resources.
        autoreleasepool(|| {
            let setup_start = Instant::now();
            let source_bytes = source_coefficients.max(1) * size_of::<Fp128Limbs>();
            let source_buffer = self.shared_buffer(source_bytes)?;
            let mut destination = source_buffer.contents().cast::<Fp128Limbs>();
            for source in sources {
                for ring in *source {
                    for &value in ring.coefficients() {
                        // SAFETY: the checked sum of all distinct source extents bounds
                        // this disjoint sequential initialization; borrowed sources stay
                        // alive through synchronous command completion below.
                        unsafe {
                            destination.write(Fp128Limbs::from_field(value));
                            destination = destination.add(1);
                        }
                    }
                }
            }
            let terms_buffer = if terms.is_empty() {
                self.shared_buffer_from_slice(&[SparseHighHalfTerm::default()])?
            } else {
                self.shared_buffer_from_slice(terms)?
            };
            let rows_buffer = self.shared_buffer_from_slice(rows)?;
            let subtract_buffer = self.shared_buffer_from_slice(subtract)?;
            let output_bytes = output_coefficients * size_of::<Fp128Limbs>();
            let output_buffer = self.shared_buffer(output_bytes)?;
            let params = SparseHighHalfParams {
                ring_d: D as u64,
                batch_count: rows.len() as u64,
                term_count: terms.len() as u64,
            };
            let allocation_bytes = [
                source_bytes,
                terms_buffer.length() as usize,
                rows_buffer.length() as usize,
                output_bytes,
                output_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "sparse high-half allocation bytes",
                    ))
            })?;
            let buffer_setup = setup_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita batched sparse ordinary-product high halves");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.sparse_high_half_pipeline);
            for (index, buffer) in [
                &source_buffer,
                &terms_buffer,
                &rows_buffer,
                &subtract_buffer,
                &output_buffer,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(output_coefficients as u64, 1, 1),
                MTLSize::new(THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: exactly one lane in each completed group initialized its output.
            let coefficients = unsafe {
                std::slice::from_raw_parts(
                    output_buffer.contents().cast::<Fp128Limbs>(),
                    output_coefficients,
                )
                .to_vec()
            };
            Ok(SparseHighHalfOutcome {
                coefficients,
                allocation_bytes,
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
