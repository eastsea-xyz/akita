//! Batched exact sparse ordinary-product high halves for relation quotients.

use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::RingSwitchRelationView;
use akita_prover::compute::SparseChallenge;
use akita_prover::compute::{CpuBackend, RingSwitchRelationKernel, SparseHighHalfBatch};
use std::time::Instant;

use crate::field::{Fp128Limbs, MetalField, F};
use crate::runtime::sparse_high_half::{SparseHighHalfRow, SparseHighHalfTerm};
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

#[cfg(test)]
mod tests;

fn cpu<const D: usize>(
    backend: &MetalBackend,
    challenges: &[SparseChallenge],
    batches: &[SparseHighHalfBatch<'_, F, D>],
    work: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    backend
        .record_opening_cpu_fallback(work)
        .map_err(MetalCommitError::into_akita)?;
    <CpuBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(
        &backend.cpu_backend(), challenges, batches,
    )
}

impl MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalSparseHighHalf::batch")]
    pub(crate) fn sparse_high_half_batch<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        batches: &[SparseHighHalfBatch<'_, F, D>],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let preparation_start = Instant::now();
        // Validate every challenge before either policy routing or GPU resources.
        // The sampler's uniqueness check is deliberately not used: repeated
        // positions and zero coefficients retain ordinary-product semantics.
        if D == 0
            || challenges.iter().any(|challenge| {
                challenge.positions.len() != challenge.coeffs.len()
                    || challenge
                        .positions
                        .iter()
                        .any(|&position| position as usize >= D)
            })
        {
            return Err(AkitaError::InvalidInput(
                "invalid sparse high-half challenge geometry".into(),
            ));
        }
        let term_count = challenges.iter().try_fold(0usize, |total, challenge| {
            total.checked_add(challenge.positions.len()).ok_or_else(|| {
                AkitaError::InvalidInput("sparse high-half term extent overflow".into())
            })
        })?;
        let output_coefficients = batches.len().checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("sparse high-half output extent overflow".into())
        })?;
        let work = term_count.saturating_mul(output_coefficients);
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let admitted = matches!(D, 64 | 128 | 256 | 512);
        if self.policy() == MetalExecutionPolicy::PreferMetal && (!admitted || work < (1 << 18)) {
            return cpu(self, challenges, batches, work);
        }
        if !admitted {
            return Err(MetalCommitError::UnsupportedShape(
                "sparse high-half ring dimension".into(),
            )
            .into_akita());
        }
        let Some(runtime) = self.runtime() else {
            return if self.policy() == MetalExecutionPolicy::PreferMetal {
                cpu(self, challenges, batches, work)
            } else {
                Err(MetalCommitError::DeviceUnavailable.into_akita())
            };
        };
        let terms = challenges
            .iter()
            .enumerate()
            .flat_map(|(index, challenge)| {
                challenge
                    .positions
                    .iter()
                    .zip(challenge.coeffs.iter())
                    .filter_map(move |(&shift, &coefficient)| {
                        (shift != 0 && coefficient != 0).then_some(SparseHighHalfTerm {
                            challenge_index: index as u64,
                            shift,
                            coefficient: i32::from(coefficient),
                        })
                    })
            })
            .collect::<Vec<_>>();
        // Each borrowed slice stays alive through synchronous dispatch. Shared
        // base pointers reuse the largest provided valid prefix; a partial view
        // still retains its own live-challenge count rather than that prefix's.
        let mut sources: Vec<&[CyclotomicRing<F, D>]> = Vec::new();
        let mut selections = Vec::with_capacity(batches.len());
        for batch in batches {
            let live = if batch.first_ring >= batch.rings.len() {
                0
            } else {
                (batch.rings.len() - 1 - batch.first_ring)
                    .checked_div(batch.ring_stride)
                    .map_or(challenges.len(), |last| challenges.len().min(last + 1))
            };
            let source = if live == 0 {
                None
            } else {
                let index = sources
                    .iter()
                    .position(|source| source.as_ptr() == batch.rings.as_ptr());
                Some(match index {
                    Some(index) => {
                        if batch.rings.len() > sources[index].len() {
                            sources[index] = batch.rings;
                        }
                        index
                    }
                    None => {
                        sources.push(batch.rings);
                        sources.len() - 1
                    }
                })
            };
            selections.push((source, live));
        }
        let mut source_offsets = Vec::with_capacity(sources.len());
        let mut source_coefficients = 0usize;
        for source in &sources {
            source_offsets.push(source_coefficients);
            source_coefficients = source
                .len()
                .checked_mul(D)
                .and_then(|count| source_coefficients.checked_add(count))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("sparse high-half source extent overflow".into())
                })?;
        }
        let mut rows = Vec::with_capacity(batches.len());
        let mut subtract = Vec::with_capacity(output_coefficients);
        for (batch, (source, live)) in batches.iter().zip(selections) {
            let source_start = source.map(|index| source_offsets[index]).unwrap_or(0);
            let source_end = if source.is_some() {
                batch
                    .rings
                    .len()
                    .checked_mul(D)
                    .and_then(|len| source_start.checked_add(len))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("sparse high-half source bound overflow".into())
                    })?
            } else {
                0
            };
            rows.push(SparseHighHalfRow {
                source_start: source_start as u64,
                source_end: source_end as u64,
                first_ring: if live == 0 {
                    0
                } else {
                    batch.first_ring as u64
                },
                ring_stride: batch.ring_stride as u64,
                live_challenges: live as u64,
            });
            for coefficient in 0..D {
                subtract.push(
                    batch
                        .subtract
                        .map(|ring| Fp128Limbs::from_field(ring.coefficients()[coefficient]))
                        .unwrap_or_default(),
                );
            }
        }
        if let Err(error) = runtime.validate_sparse_high_half_buffers(
            source_coefficients,
            terms.len(),
            rows.len(),
            D,
        ) {
            return if self.policy() == MetalExecutionPolicy::PreferMetal {
                cpu(self, challenges, batches, work)
            } else {
                Err(error.into_akita())
            };
        }
        let preparation_time = preparation_start.elapsed();
        // Once dispatched, every failure propagates; no post-submission CPU replay.
        let output = runtime
            .sparse_high_half_batch::<D>(&sources, challenges.len(), &terms, &rows, &subtract)
            .map_err(MetalCommitError::into_akita)?;
        if output.coefficients.len() != output_coefficients {
            return Err(AkitaError::InvalidSize {
                expected: output_coefficients,
                actual: output.coefficients.len(),
            });
        }
        let readback_start = Instant::now();
        let coefficients = output
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        let result = coefficients
            .as_chunks::<D>()
            .0
            .iter()
            .map(|row| CyclotomicRing::from_slice(row))
            .collect();
        let canonical_readback = readback_start.elapsed();
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += output.timings.command_wall;
            metrics.gpu_active_time += output.timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += preparation_time + output.timings.buffer_setup;
            metrics.readback_time += output.timings.readback_copy + canonical_readback;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(output.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(result)
    }
}
