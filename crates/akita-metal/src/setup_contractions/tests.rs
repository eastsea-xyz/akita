use super::*;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
use akita_types::SetupMatrixCapacity;
use jolt_field::{CanonicalEncoding, Ext2, ExtField, MulBaseUnreduced, One, Ring};

fn cpu<E: FpExtEncoding<F> + MulBaseUnreduced<F>>(
    fields: &[F],
    plans: &[SetupColumnContractionPlan<'_, E>],
) -> Vec<Vec<E>> {
    plans
        .iter()
        .map(|plan| {
            let mut values = vec![E::zero(); plan.columns.len() * plan.batch_count];
            for (offset, column) in plan.columns.clone().enumerate() {
                for (row, weights) in plan.row_weights {
                    let start = (row * plan.matrix_columns + column) * plan.ring_d;
                    let evaluated = eval_flat_ring_at_pows_fast(
                        &fields[start..start + plan.ring_d],
                        plan.alpha_powers,
                    );
                    for (batch, &weight) in weights.iter().enumerate() {
                        values[offset * plan.batch_count + batch] += weight * evaluated;
                    }
                }
            }
            values
        })
        .collect()
}

#[test]
fn scalar_setup_contractions_random_all_dimensions_and_batched_role_shapes() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        18,
        1,
        SetupMatrixCapacity {
            num_field_elements: 5 * 11 * 512,
        },
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let fields = setup.expanded.shared_matrix().as_field_slice();
    let mut state = 0x2048_aabb_aa57_0123u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let lo = state;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        F::from_u128_reduced(u128::from(lo) | (u128::from(state) << 64))
    };
    for dimension in [64, 128, 256, 512] {
        let alpha = random();
        let a = scalar_powers(alpha, dimension);
        let b = scalar_powers(random(), 64);
        let d = scalar_powers(random(), 128);
        let a_weights = vec![
            (4, vec![random(), random(), F::zero()]),
            (1, vec![random(), F::zero(), random()]),
            (4, vec![random(), random(), random()]),
        ];
        let b_weights = vec![
            (2, vec![random(), random()]),
            (0, vec![F::zero(), random()]),
        ];
        let d_weights = vec![(1, vec![random()]), (3, vec![random()])];
        let plans = [
            SetupColumnContractionPlan {
                matrix_rows: 5,
                matrix_columns: 11,
                ring_d: dimension,
                columns: 3..10,
                row_weights: &a_weights,
                alpha_powers: &a,
                batch_count: 3,
            },
            SetupColumnContractionPlan {
                matrix_rows: 3,
                matrix_columns: 7,
                ring_d: 64,
                columns: 1..6,
                row_weights: &b_weights,
                alpha_powers: &b,
                batch_count: 2,
            },
            SetupColumnContractionPlan {
                matrix_rows: 4,
                matrix_columns: 9,
                ring_d: 128,
                columns: 2..9,
                row_weights: &d_weights,
                alpha_powers: &d,
                batch_count: 1,
            },
        ];
        let expected = cpu(fields, &plans);
        for _ in 0..2 {
            backend.begin_opening_metrics().unwrap();
            assert_eq!(
                backend
                    .contract_setup_columns(&prepared, &plans)
                    .unwrap()
                    .unwrap(),
                expected
            );
            let metrics = backend.last_opening_metrics().unwrap().unwrap();
            assert_eq!(metrics.cpu_fallback_calls, 0);
            assert!(metrics.command_wall_time > std::time::Duration::ZERO);
        }
    }
}

#[test]
fn scalar_setup_contractions_zero_empty_and_invalid_batch_admission() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        12,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 5 * 128,
        },
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let powers = scalar_powers(F::from_u64(3), 128);
    let weights = vec![(2, vec![F::zero(); 2])];
    let valid = SetupColumnContractionPlan {
        matrix_rows: 3,
        matrix_columns: 5,
        ring_d: 128,
        columns: 1..4,
        row_weights: &weights,
        alpha_powers: &powers,
        batch_count: 2,
    };
    let wrong_row = vec![(3, vec![F::one(); 2])];
    let wrong_batches = vec![(1, vec![F::one()])];
    for bad in [
        SetupColumnContractionPlan {
            columns: 1..6,
            ..valid.clone()
        },
        SetupColumnContractionPlan {
            row_weights: &wrong_row,
            ..valid.clone()
        },
        SetupColumnContractionPlan {
            row_weights: &wrong_batches,
            ..valid.clone()
        },
        SetupColumnContractionPlan {
            alpha_powers: &powers[..127],
            ..valid.clone()
        },
        SetupColumnContractionPlan {
            matrix_rows: usize::MAX,
            ..valid.clone()
        },
        SetupColumnContractionPlan {
            columns: 1..1,
            batch_count: 0,
            ..valid.clone()
        },
    ] {
        backend.begin_opening_metrics().unwrap();
        assert!(backend
            .contract_setup_columns(&prepared, &[valid.clone(), bad])
            .is_err());
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 0);
        let metrics = backend.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
        assert_eq!(metrics.cpu_fallback_calls, 0);
    }
    assert_eq!(
        backend
            .contract_setup_columns(&prepared, std::slice::from_ref(&valid))
            .unwrap()
            .unwrap(),
        vec![vec![F::zero(); 6]]
    );
    let no_rows = SetupColumnContractionPlan {
        matrix_rows: 0,
        row_weights: &[],
        ..valid.clone()
    };
    assert_eq!(
        backend
            .contract_setup_columns(&prepared, &[no_rows])
            .unwrap()
            .unwrap(),
        vec![vec![F::zero(); 6]]
    );
    assert_eq!(
        backend
            .contract_setup_columns(
                &prepared,
                &[SetupColumnContractionPlan {
                    columns: 2..2,
                    ..valid
                }]
            )
            .unwrap()
            .unwrap(),
        vec![Vec::<F>::new()]
    );
    assert!(backend
        .contract_setup_columns::<F>(&prepared, &[])
        .unwrap()
        .unwrap()
        .is_empty());
}

#[test]
fn setup_contractions_proper_degree_two_selects_exact_cpu_oracle() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        12,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 5 * 128,
        },
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let powers = scalar_powers(
        Ext2::<F>::from_base_slice(&[F::from_u64(3), F::from_u64(7)]),
        128,
    );
    let weights = vec![(
        2,
        vec![Ext2::from_base_slice(&[F::from_u64(11), F::from_u64(13)])],
    )];
    let plans = [SetupColumnContractionPlan {
        matrix_rows: 3,
        matrix_columns: 5,
        ring_d: 128,
        columns: 1..4,
        row_weights: &weights,
        alpha_powers: &powers,
        batch_count: 1,
    }];
    backend.begin_opening_metrics().unwrap();
    assert!(backend
        .contract_setup_columns(&prepared, &plans)
        .unwrap()
        .is_none());
    assert_eq!(
        backend
            .last_opening_metrics()
            .unwrap()
            .unwrap()
            .command_wall_time,
        std::time::Duration::ZERO
    );
    let backend_cpu = CpuBackend::DEFAULT;
    let prepared_cpu = backend_cpu.prepare_setup(&setup).unwrap();
    assert!(backend_cpu
        .contract_setup_columns(&prepared_cpu, &plans)
        .unwrap()
        .is_none());
    let expected = cpu(setup.expanded.shared_matrix().as_field_slice(), &plans);
    assert!(
        expected[0]
            .iter()
            .any(|value| value.degree_one_base().is_none()),
        "the fallback oracle must retain proper-extension coordinates"
    );
}
