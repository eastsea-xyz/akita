use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_prover::compute::SetupColumnContractionPlan;
use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
use akita_types::SetupMatrixCapacity;
use jolt_field::{Ring, Zero};

#[test]
fn scalar_setup_columns_require_metal_hook_and_match_cpu_reference() {
    const D: usize = 128;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        12,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 5 * D,
        },
        None,
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let powers = scalar_powers(F::from_u64(3), D);
    let row_weights = vec![
        (2, vec![F::from_u64(7), F::from_i64(-3)]),
        (0, vec![F::from_u64(11), F::zero()]),
        (2, vec![F::from_u64(5), F::from_u64(13)]),
    ];
    let plan = SetupColumnContractionPlan {
        matrix_rows: 3,
        matrix_columns: 5,
        ring_d: D,
        columns: 1..4,
        row_weights: &row_weights,
        alpha_powers: &powers,
        batch_count: 2,
    };
    let fields = setup.expanded.shared_matrix().as_field_slice();
    let mut expected = vec![F::zero(); 6];
    for (offset, column) in (1..4).enumerate() {
        for (row, weights) in &row_weights {
            let start = (row * 5 + column) * D;
            let value = eval_flat_ring_at_pows_fast(&fields[start..start + D], &powers);
            for (batch, &weight) in weights.iter().enumerate() {
                expected[offset * 2 + batch] += weight * value;
            }
        }
    }
    backend.begin_opening_metrics().unwrap();
    let result = backend
        .contract_setup_columns(&prepared, std::slice::from_ref(&plan))
        .unwrap()
        .expect("RequireMetal scalar setup contraction must execute its optional GPU hook");
    assert_eq!(result, vec![expected]);
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
    assert!(
        metrics.command_wall_time > std::time::Duration::ZERO,
        "scalar setup contractions must submit GPU work"
    );
}
