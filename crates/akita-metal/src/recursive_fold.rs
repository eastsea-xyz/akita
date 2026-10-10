use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::{RecursiveFoldView, SuffixWitnessView};
use akita_prover::compute::OpeningFoldPlan;
use akita_types::AkitaExpandedSetup;
use jolt_field::{One, Zero};
use std::sync::Arc;
use std::time::Instant;

use crate::field::{Fp128Limbs, MetalField, F};
use crate::runtime::{
    RecursiveFoldDispatchOutcome, RecursiveFoldParams, RecursiveFoldSource, RecursiveSparseTerm,
    RecursiveWeightTerm,
};
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy, MetalPreparedSetup};

mod kernels;
#[cfg(test)]
mod tests;

enum FoldInput<'a> {
    Packed {
        owner: Arc<[u8]>,
    },
    Field {
        fields: &'a [F],
        expanded: &'a AkitaExpandedSetup<F>,
    },
}

fn suffix_input<const D: usize>(
    view: SuffixWitnessView<'_, F, D>,
    positions: usize,
) -> Result<(FoldInput<'_>, RecursiveFoldParams), AkitaError> {
    let (owner, width, stored, physical, live_rings) = view.packed_fold_storage()?;
    let blocks = view.num_live_blocks(positions)?;
    Ok((
        FoldInput::Packed { owner },
        RecursiveFoldParams {
            stored_coefficients: stored as u64,
            physical_coefficients: physical as u64,
            live_rings: live_rings as u64,
            positions_per_block: positions as u64,
            num_blocks: blocks as u64,
            ring_d: D as u64,
            bit_width: u64::from(width),
            num_digits: 1,
            log_basis: 1,
            threshold: Fp128Limbs::default(),
        },
    ))
}

fn recursive_input<const D: usize>(
    view: RecursiveFoldView<'_, F, D>,
    positions: usize,
) -> Result<(FoldInput<'_>, RecursiveFoldParams), AkitaError> {
    match view {
        RecursiveFoldView::Witness(view) => suffix_input(view, positions),
        RecursiveFoldView::SetupPrefix { expanded, .. } => {
            let (fields, expected_positions, blocks) = view
                .setup_prefix_fold_coefficients()?
                .ok_or(AkitaError::InvalidProof)?;
            if positions != expected_positions {
                return Err(AkitaError::InvalidSetup(
                    "recursive prefix positions disagree with frozen geometry".into(),
                ));
            }
            Ok((
                FoldInput::Field { fields, expanded },
                RecursiveFoldParams {
                    stored_coefficients: fields.len() as u64,
                    physical_coefficients: fields.len() as u64,
                    live_rings: (fields.len() / D) as u64,
                    positions_per_block: positions as u64,
                    num_blocks: blocks as u64,
                    ring_d: D as u64,
                    bit_width: 0,
                    num_digits: 1,
                    log_basis: 1,
                    threshold: Fp128Limbs::default(),
                },
            ))
        }
    }
}

fn fold_weights<const D: usize>(
    plan: OpeningFoldPlan<'_, F>,
    blocks: usize,
) -> Result<(usize, Vec<RecursiveWeightTerm>, Vec<RecursiveWeightTerm>), AkitaError> {
    let terms = |rings: Vec<CyclotomicRing<F, D>>| {
        rings
            .into_iter()
            .enumerate()
            .flat_map(|(position, ring)| {
                ring.coeffs
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| !value.is_zero())
                    .map(move |(shift, weight)| RecursiveWeightTerm {
                        weight: weight.into_device(),
                        position: position as u64,
                        shift: shift as u64,
                    })
            })
            .collect::<Vec<_>>()
    };
    match plan {
        OpeningFoldPlan::Base {
            live_block_weights,
            position_weights,
            num_positions_per_block,
        } => {
            if !num_positions_per_block.is_power_of_two()
                || blocks == 0
                || live_block_weights.len() != blocks
                || position_weights.len() != num_positions_per_block
            {
                return Err(AkitaError::InvalidInput(
                    "opening fold weights do not match exact L/F geometry".into(),
                ));
            }
            let scalars = |values: &[F]| {
                values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| !value.is_zero())
                    .map(|(position, weight)| RecursiveWeightTerm {
                        weight: weight.into_device(),
                        position: position as u64,
                        shift: 0,
                    })
                    .collect()
            };
            Ok((
                num_positions_per_block,
                scalars(position_weights),
                scalars(live_block_weights),
            ))
        }
        OpeningFoldPlan::Subfield {
            multipliers,
            num_positions_per_block,
        } => {
            multipliers.ensure_ring_dim::<D>()?;
            if !num_positions_per_block.is_power_of_two()
                || blocks == 0
                || multipliers.position_len() != num_positions_per_block
                || multipliers.fold_len() != blocks
            {
                return Err(AkitaError::InvalidInput(
                    "opening fold weights do not match exact L/F geometry".into(),
                ));
            }
            Ok((
                num_positions_per_block,
                terms(multipliers.materialize_position_rings::<D>()?),
                terms(multipliers.materialize_fold_rings::<D>()?),
            ))
        }
    }
}

fn fold_positions(plan: OpeningFoldPlan<'_, F>) -> usize {
    match plan {
        OpeningFoldPlan::Base {
            num_positions_per_block,
            ..
        }
        | OpeningFoldPlan::Subfield {
            num_positions_per_block,
            ..
        } => num_positions_per_block,
    }
}

fn sparse_terms<const D: usize>(
    params: &mut RecursiveFoldParams,
    plan: akita_prover::compute::DecomposeFoldPlan<'_>,
) -> Result<Vec<RecursiveSparseTerm>, AkitaError> {
    if plan.challenges.len() != params.num_blocks as usize {
        return Err(AkitaError::InvalidSize {
            expected: params.num_blocks as usize,
            actual: plan.challenges.len(),
        });
    }
    if params.bit_width != 0 && plan.num_digits != 1 {
        return Err(AkitaError::InvalidSetup(
            "recursive digit witness decomposition requires one tight digit".into(),
        ));
    }
    if params.bit_width == 0 {
        if plan.num_digits == 0 || !(1..=16).contains(&plan.log_basis) {
            return Err(MetalCommitError::UnsupportedShape(
                "recursive field decomposition requires a basis up to i16".into(),
            )
            .into_akita());
        }
        let modulus = (-F::one()).to_canonical_u128() + 1;
        params.num_digits = plan.num_digits as u64;
        params.log_basis = u64::from(plan.log_basis);
        params.threshold = Fp128Limbs::from_u128(decompose_centering_threshold(
            plan.num_digits,
            plan.log_basis,
            modulus,
        ));
    }
    let mut terms = Vec::new();
    for (block, challenge) in plan.challenges.iter().enumerate() {
        if challenge.positions.len() != challenge.coeffs.len()
            || challenge.positions.iter().any(|&p| p as usize >= D)
        {
            return Err(AkitaError::InvalidInput(
                "recursive sparse challenge shape".into(),
            ));
        }
        if akita_prover::backend::poly_helpers::should_use_rotated_challenge::<D>(challenge) {
            // The exact CPU D64 rotated route assigns dense[position] = coeff:
            // only the final coefficient at a repeated position contributes.
            let mut dense = [0i8; 64];
            for (&shift, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                dense[shift as usize] = coefficient;
            }
            terms.extend(
                dense
                    .into_iter()
                    .enumerate()
                    .filter(|(_, coefficient)| *coefficient != 0)
                    .map(|(shift, coefficient)| RecursiveSparseTerm {
                        block: block as u64,
                        shift: shift as u32,
                        coefficient: i32::from(coefficient),
                    }),
            );
        } else {
            terms.extend(challenge.positions.iter().zip(&challenge.coeffs).map(
                |(&shift, &coefficient)| RecursiveSparseTerm {
                    block: block as u64,
                    shift,
                    coefficient: i32::from(coefficient),
                },
            ));
        }
    }
    Ok(terms)
}

impl MetalBackend {
    fn recursive_fold_supported<const D: usize>(
        &self,
        field_basis: Option<u32>,
    ) -> Result<bool, AkitaError> {
        if field_basis
            .is_some_and(|bits| akita_types::SignedDigitKernel::for_log_basis(bits).is_none())
        {
            return Err(AkitaError::InvalidSetup(
                "recursive field decomposition requires a valid i8/i16 basis".into(),
            ));
        }
        if self.runtime().is_some() && matches!(D, 64 | 128 | 256 | 512) {
            return Ok(true);
        }
        match self.policy() {
            MetalExecutionPolicy::PreferMetal => Ok(false),
            MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                "recursive folding requires Metal, D64/D128/D256/D512 and a field basis up to i16"
                    .into(),
            )
            .into_akita()),
        }
    }

    fn prepare_fold_input(
        &self,
        prepared: Option<&MetalPreparedSetup>,
        input: FoldInput<'_>,
        p: RecursiveFoldParams,
    ) -> Result<RecursiveFoldSource, AkitaError> {
        match input {
            FoldInput::Packed { owner } => Ok(RecursiveFoldSource::Packed { owner }),
            FoldInput::Field { fields, expanded } => {
                let runtime = self
                    .runtime()
                    .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
                let buffer = if let Some(prepared) =
                    prepared.filter(|p| std::ptr::eq(p.expanded.as_ref(), expanded))
                {
                    let matrix = prepared.matrix(
                        runtime,
                        p.ring_d as usize,
                        1,
                        fields.len() / p.ring_d as usize,
                    )?;
                    self.update_opening_metrics(|m| {
                        m.buffer_setup_time += matrix.prepare_time;
                        if !matrix.cache_hit {
                            m.allocation_bytes = m.allocation_bytes.saturating_add(matrix.bytes);
                        }
                    })
                    .map_err(MetalCommitError::into_akita)?;
                    matrix.buffer.as_ref().clone()
                } else {
                    let start = Instant::now();
                    // Copy canonical field storage, retaining no borrowed no-copy buffer after return.
                    let buffer = runtime
                        .shared_buffer_from_slice(fields)
                        .map_err(MetalCommitError::into_akita)?;
                    self.update_opening_metrics(|m| {
                        m.buffer_setup_time += start.elapsed();
                        m.allocation_bytes = m
                            .allocation_bytes
                            .saturating_add(std::mem::size_of_val(fields));
                    })
                    .map_err(MetalCommitError::into_akita)?;
                    buffer
                };
                Ok(RecursiveFoldSource::Field { buffer })
            }
        }
    }

    fn record_recursive_fold(
        &self,
        outcome: &RecursiveFoldDispatchOutcome,
    ) -> Result<(), AkitaError> {
        self.update_opening_metrics(|m| {
            m.buffer_setup_time += outcome.timings.buffer_setup;
            m.command_wall_time += outcome.timings.command_wall;
            m.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
            m.readback_time += outcome.timings.readback_copy;
            m.allocation_bytes = m.allocation_bytes.saturating_add(outcome.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)
    }
}
