use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_challenges::SparseChallenge;
use akita_prover::backend::RecursiveWitnessFlat;
use akita_prover::compute::{CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, OpeningFoldPlan};
use jolt_field::{One, Ring};

#[test]
fn recursive_evaluate_require_metal_rejects_unsupported_dimension_before_cpu_work() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    backend.begin_opening_metrics().unwrap();
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 32]);
    let weights = [F::one()];
    let result = OpeningFoldKernel::evaluate_and_fold(
        &backend,
        None,
        witness.view::<F, 32>().unwrap(),
        OpeningFoldPlan::Base {
            live_block_weights: &weights,
            position_weights: &weights,
            num_positions_per_block: 1,
        },
    );
    assert!(
        result.is_err(),
        "D32 must be rejected before silently executing its CPU fold"
    );
    assert_eq!(
        backend
            .last_opening_metrics()
            .unwrap()
            .unwrap()
            .cpu_fallback_calls,
        0
    );
}

#[test]
fn recursive_sparse_require_metal_rejects_unsupported_dimension_before_cpu_work() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    backend.begin_opening_metrics().unwrap();
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 32]);
    let challenges = [SparseChallenge {
        positions: [31].into_iter().collect(),
        coeffs: [-1].into_iter().collect(),
    }];
    let plan = DecomposeFoldPlan {
        challenges: &challenges,
        num_positions_per_block: 1,
        num_digits: 1,
        log_basis: 0,
    };
    let result =
        OpeningFoldKernel::decompose_fold(&backend, None, witness.view::<F, 32>().unwrap(), plan);
    assert!(
        result.is_err(),
        "D32 must be rejected before silently executing its CPU fold"
    );
    assert_eq!(
        backend
            .last_opening_metrics()
            .unwrap()
            .unwrap()
            .cpu_fallback_calls,
        0
    );
}

#[test]
fn recursive_supported_fold_executes_on_metal_and_matches_cpu() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    backend.begin_opening_metrics().unwrap();
    let witness =
        RecursiveWitnessFlat::from_i8_digits((0..896).map(|i| (i % 7) as i8 - 3).collect());
    let positions = [2, 3, 5, 7].map(F::from_u64);
    let folds = [11, 13].map(F::from_u64);
    let plan = OpeningFoldPlan::Base {
        live_block_weights: &folds,
        position_weights: &positions,
        num_positions_per_block: 4,
    };
    let view = witness.view::<F, 128>().unwrap();
    let expected =
        OpeningFoldKernel::evaluate_and_fold(&CpuBackend::DEFAULT, None, view, plan).unwrap();
    let actual = OpeningFoldKernel::evaluate_and_fold(&backend, None, view, plan).unwrap();
    assert_eq!(actual, expected);
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    assert_eq!(
        metrics.cpu_fallback_calls, 0,
        "supported recursive folds must not delegate to CPU"
    );
    assert!(metrics.command_wall_time > std::time::Duration::ZERO);
}
