use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_error::{checked, AkitaError};
use akita_prover::backend::DenseView;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, SuffixWitnessView};
use akita_types::{balanced_signed_digit_abs_bound, RingVec, MAX_I8_LOG_BASIS};
use jolt_field::One;

use crate::field::F;
use crate::runtime::{
    MetalRuntime, PackedRecursiveCommitParams, RecursiveCommitInput, RecursiveCommitParams,
    FP128_D512_LINEAR_RELATION_NUM_PRIMES,
};
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

// Dense advice commitments deliberately retain the CPU implementation at any
// ring dimension. This compatibility route is counted as CPU source work in
// the mixed opening stack, including when Metal is required for suffix witnesses.
impl<const D: usize> RootCommitKernel<DenseView<'_, F, D>, F, D> for MetalBackend {
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<DenseView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let source_count = sources.len();
        let result = self
            .cpu_backend()
            .commit_inner_group(&prepared.cpu, sources, plan)?;
        self.record_opening_cpu_fallback(source_count)
            .map_err(MetalCommitError::into_akita)?;
        Ok(result)
    }
}

impl<const D: usize> RootCommitKernel<SuffixWitnessView<'_, F, D>, F, D> for MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalBackend::recursive_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<SuffixWitnessView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let Some(balanced_bound) = balanced_signed_digit_abs_bound(plan.log_basis_inner) else {
            return Err(AkitaError::InvalidSetup(
                "invalid recursive commitment basis".into(),
            ));
        };
        if plan.n_a == 0
            || plan.num_positions_per_block == 0
            || plan.num_digits_inner == 0
            || checked::product([plan.num_digits_inner, plan.log_basis_inner as usize])
                .is_none_or(|bits| bits > 128 + plan.log_basis_inner as usize)
        {
            return Err(AkitaError::InvalidSetup(
                "invalid recursive commitment geometry".into(),
            ));
        }
        if !matches!(D, 64 | 128 | 256 | 512)
            || plan.log_basis_inner > MAX_I8_LOG_BASIS
            || self.runtime().is_none()
        {
            return match self.policy() {
                MetalExecutionPolicy::PreferMetal => {
                    let source_count = sources.len();
                    let output = self.cpu_backend().commit_inner_group(&prepared.cpu, sources, plan)?;
                    self.record_opening_cpu_fallback(source_count).map_err(MetalCommitError::into_akita)?;
                    Ok(output)
                }
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::UnsupportedShape(format!(
                        "recursive commitment requires Metal, D64/D128/D256/D512 and an i8 basis; D={D}"
                    ))
                    .into_akita())
                }
            };
        }
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let cols = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(|| MetalCommitError::ShapeOverflow("recursive columns").into_akita())?;
        let matrix_rings = checked::product([plan.n_a, cols]).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("recursive matrix rings").into_akita()
        })?;
        // A single plane uses the stored raw bytes, even outside its declared
        // balanced interval. Multiple planes use the CPU's canonical decomposition.
        let rhs_abs_bound = if plan.num_digits_inner == 1 {
            128
        } else {
            balanced_bound
        };
        let blocks_per_group = MetalRuntime::recursive_commit_blocks_per_group::<D>();
        let mut parameters = Vec::with_capacity(sources.len());
        let mut inputs = Vec::with_capacity(sources.len());
        let mut supported = true;
        for source in &sources {
            let blocks = source.num_live_blocks(plan.num_positions_per_block)?;
            let (owner, width, stored, physical, _) = source.packed_fold_storage()?;
            let minimum_rings =
                checked::mul_add(blocks.saturating_sub(1), plan.num_positions_per_block, 1)
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("recursive live extent").into_akita()
                    })?;
            if blocks == 0 || !physical.is_multiple_of(D) || physical / D < minimum_rings {
                return Err(AkitaError::InvalidSetup(
                    "recursive witness does not cover its live blocks".into(),
                ));
            }
            supported &=
                runtime.supports_fp128_recursive_commit::<D>(blocks, plan.n_a, cols, rhs_abs_bound);
            let output = checked::product([blocks, plan.n_a, D])
                .ok_or_else(|| MetalCommitError::ShapeOverflow("recursive output").into_akita())?;
            let params = RecursiveCommitParams {
                num_blocks: blocks as u64,
                blocks_per_group: blocks_per_group as u64,
                num_block_groups: blocks.div_ceil(blocks_per_group) as u64,
                num_rows: plan.n_a as u64,
                num_cols: cols as u64,
                ring_d: D as u64,
                num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
                matrix_rings: matrix_rings as u64,
                output_coefficients: output as u64,
                rhs_abs_bound,
            };
            let capacity =
                checked::product([blocks, plan.num_positions_per_block, D]).ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("recursive raw source capacity").into_akita()
                })?;
            let input = RecursiveCommitInput::Packed {
                owner,
                params: PackedRecursiveCommitParams {
                    stored_coefficients: stored.min(capacity) as u64,
                    physical_coefficients: physical.min(capacity) as u64,
                    ring_d: D as u64,
                    bit_width: u64::from(width),
                    num_digits: plan.num_digits_inner as u64,
                    log_basis: u64::from(plan.log_basis_inner),
                    threshold: crate::field::Fp128Limbs::from_u128(decompose_centering_threshold(
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                        (-F::one()).to_canonical_u128() + 1,
                    )),
                },
            };
            match runtime.validate_recursive_commit_input::<D>(&input, params) {
                Ok(()) => {}
                Err(
                    MetalCommitError::UnsupportedShape(_) | MetalCommitError::BufferTooLong { .. },
                ) => supported = false,
                Err(error) => return Err(error.into_akita()),
            }
            inputs.push(input);
            parameters.push(params);
        }
        if !supported {
            return match self.policy() {
                MetalExecutionPolicy::PreferMetal => {
                    let source_count = sources.len();
                    let output =
                        self.cpu_backend()
                            .commit_inner_group(&prepared.cpu, sources, plan)?;
                    self.record_opening_cpu_fallback(source_count)
                        .map_err(MetalCommitError::into_akita)?;
                    Ok(output)
                }
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "recursive commitment exceeds exact CRT or device bounds".into(),
                )
                .into_akita()),
            };
        }
        // Validate every source before preparing matrices or launching any GPU work.
        let matrix = prepared.recursive_matrix::<D>(runtime, parameters[0])?;
        self.update_opening_metrics(|metrics| {
            metrics.buffer_setup_time += matrix.prepare_time;
            if !matrix.cache_hit {
                metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(matrix.bytes);
            }
        })
        .map_err(MetalCommitError::into_akita)?;
        inputs
            .into_iter()
            .zip(parameters)
            .map(|(input, params)| {
                let outcome = runtime
                    .dispatch_fp128_recursive_commit::<D>(&matrix.view(), input, params)
                    .map_err(MetalCommitError::into_akita)?;
                self.update_opening_metrics(|metrics| {
                    metrics.buffer_setup_time += outcome.timings.buffer_setup;
                    metrics.command_wall_time += outcome.timings.command_wall;
                    metrics.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
                    metrics.readback_time += outcome.timings.readback_copy;
                    metrics.allocation_bytes = metrics
                        .allocation_bytes
                        .saturating_add(outcome.allocation_bytes);
                })
                .map_err(MetalCommitError::into_akita)?;
                let coefficients = outcome
                    .coefficients
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            .into_field(index)
                            .map_err(MetalCommitError::into_akita)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CommitInnerWitness {
                    inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)?,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, DensePoly, RecursiveWitnessFlat,
        RootCommitSource, SuffixWitnessView,
    };
    use akita_types::SetupMatrixCapacity;
    use jolt_field::Ring;

    use crate::field::F;
    use crate::{profiling, MetalBackend, MetalExecutionPolicy};

    fn random_witness<const D: usize>(rings: usize, seed: u64) -> RecursiveWitnessFlat {
        let mut state = seed;
        RecursiveWitnessFlat::from_i8_digits(
            (0..rings * D)
                .map(|index| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match index % 11 {
                        0 => i8::MIN,
                        1 => i8::MAX,
                        _ => (state >> 32) as i8,
                    }
                })
                .collect(),
        )
    }

    fn assert_random_parity<const D: usize>(num_digits_inner: usize) {
        let plan = CommitInnerPlan {
            n_a: 3,
            num_positions_per_block: 4,
            num_digits_inner,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            14,
            1,
            SetupMatrixCapacity {
                num_field_elements: plan.n_a * plan.num_positions_per_block * num_digits_inner * D,
            },
            None,
        )
        .unwrap();
        let witnesses = [random_witness::<D>(19, 17), random_witness::<D>(70, 91)];
        let sources = witnesses
            .iter()
            .map(|witness| witness.view::<F, D>().unwrap())
            .collect::<Vec<_>>();
        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let expected = cpu
            .commit_inner_group(&cpu_prepared, sources.clone(), plan)
            .unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();

        profiling::set_enabled(true);
        for iteration in 0..2 {
            let before = profiling::snapshot();
            let actual = <MetalBackend as RootCommitKernel<SuffixWitnessView<'_, F, D>, F, D>>::commit_inner_group(
                &metal,
                &prepared,
                sources.clone(),
                plan,
            )
            .unwrap();
            let after = profiling::snapshot();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.inner_rows, expected.inner_rows);
            }
            assert!(after.submitted_command_buffers > before.submitted_command_buffers);
            assert_eq!(after.pending_command_buffers, 0);
            if iteration == 1 {
                // A resident transformed matrix needs only one matvec command per source.
                assert_eq!(
                    after.submitted_command_buffers - before.submitted_command_buffers,
                    sources.len() as u64
                );
            }
        }
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
        assert_eq!(
            prepared.matrix_cache_bytes().unwrap(),
            plan.n_a
                * plan.num_positions_per_block
                * num_digits_inner
                * 2
                * D
                * 6
                * size_of::<i32>()
        );
        profiling::set_enabled(false);
    }

    #[test]
    fn recursive_commit_random_d64_d128_matches_cpu_and_dispatches() {
        // Two base-8 planes truncate large digits; three preserve them with balanced carries.
        assert_random_parity::<64>(1);
        assert_random_parity::<64>(2);
        assert_random_parity::<64>(3);
        assert_random_parity::<128>(1);
        assert_random_parity::<128>(2);
        assert_random_parity::<128>(3);
    }

    #[test]
    fn recursive_commit_random_d512_matches_cpu_and_dispatches() {
        assert_random_parity::<512>(1);
        assert_random_parity::<512>(3);
    }

    #[test]
    fn recursive_commit_random_d256_matches_cpu_and_dispatches() {
        assert_random_parity::<256>(1);
        assert_random_parity::<256>(3);
    }

    #[test]
    fn recursive_commit_unsupported_dimension_respects_policy() {
        const D: usize = 1024;
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: 2,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            12,
            1,
            SetupMatrixCapacity {
                num_field_elements: 2 * D,
            },
            None,
        )
        .unwrap();
        let witness = random_witness::<D>(2, 21);
        let source = witness.view::<F, D>().unwrap();
        let cpu = CpuBackend::DEFAULT;
        let expected = cpu
            .commit_inner_group(&cpu.prepare_setup(&setup).unwrap(), vec![source], plan)
            .unwrap();
        for policy in [
            MetalExecutionPolicy::RequireMetal,
            MetalExecutionPolicy::PreferMetal,
        ] {
            let metal = MetalBackend::new(policy).unwrap();
            let prepared = metal.prepare_setup(&setup).unwrap();
            profiling::set_enabled(true);
            let before = profiling::snapshot();
            let actual = <MetalBackend as RootCommitKernel<SuffixWitnessView<'_, F, D>, F, D>>::commit_inner_group(
                &metal, &prepared, vec![source], plan,
            );
            let after = profiling::snapshot();
            profiling::set_enabled(false);
            assert_eq!(
                after.submitted_command_buffers,
                before.submitted_command_buffers
            );
            if policy == MetalExecutionPolicy::RequireMetal {
                assert!(actual.is_err());
            } else {
                assert_eq!(actual.unwrap()[0].inner_rows, expected[0].inner_rows);
            }
        }
    }

    #[test]
    fn dense_advice_commit_is_counted_as_cpu_fallback() {
        use akita_algebra::CyclotomicRing;

        const D: usize = 512;
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: 2,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            14,
            1,
            SetupMatrixCapacity {
                num_field_elements: 2 * D,
            },
            None,
        )
        .unwrap();
        let source = DensePoly::from_ring_coeffs::<D>(vec![
            CyclotomicRing::from_coefficients(
                [F::from_i8(-2); D]
            );
            4
        ]);
        let view = <DensePoly<F> as RootCommitSource<F, D>>::commit_view(&source).unwrap();
        let cpu = CpuBackend::DEFAULT;
        let expected = cpu
            .commit_inner_group(&cpu.prepare_setup(&setup).unwrap(), vec![view], plan)
            .unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .commit_inner_group(&prepared, vec![view], plan)
            .unwrap();
        assert_eq!(actual[0].inner_rows, expected[0].inner_rows);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 1);
        assert_eq!(metrics.cpu_fallback_work_units, 1);
        assert_eq!(metrics.allocation_bytes, 0);
    }
}

#[cfg(test)]
mod packed_tests;
