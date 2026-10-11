use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_prover::backend::RecursiveWitnessFlat;
use akita_prover::compute::{
    CommitInnerPlan, CpuBackend, OpeningFoldKernel, OpeningFoldPlan, RootCommitKernel,
};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
use akita_types::SetupMatrixCapacity;
use jolt_field::Ring;

#[test]
fn recursive_commit_reuses_packed_input_before_first_opening_fold() {
    const D: usize = 128;
    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: 4,
        num_digits_inner: 1,
        log_basis_inner: 3,
    };
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        12,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * D,
        },
        None,
    )
    .unwrap();
    let witness =
        RecursiveWitnessFlat::from_i8_digits((0..7 * D).map(|i| (i % 7) as i8 - 3).collect());
    let view = witness.view::<F, D>().unwrap();
    let cpu = CpuBackend::DEFAULT;
    let expected = cpu
        .commit_inner_group(&cpu.prepare_setup(&setup).unwrap(), vec![view], plan)
        .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let actual = backend
        .commit_inner_group(&prepared, vec![view], plan)
        .unwrap();
    assert_eq!(actual[0].inner_rows, expected[0].inner_rows);
    let positions = [2, 3, 5, 7].map(F::from_u64);
    let folds = [11, 13].map(F::from_u64);
    let fold = OpeningFoldPlan::Base {
        live_block_weights: &folds,
        position_weights: &positions,
        num_positions_per_block: 4,
    };
    backend.begin_opening_metrics().unwrap();
    let first = OpeningFoldKernel::evaluate_and_fold(&backend, None, view, fold).unwrap();
    let first_bytes = backend
        .last_opening_metrics()
        .unwrap()
        .unwrap()
        .allocation_bytes;
    backend.begin_opening_metrics().unwrap();
    let second = OpeningFoldKernel::evaluate_and_fold(&backend, None, view, fold).unwrap();
    let second_bytes = backend
        .last_opening_metrics()
        .unwrap()
        .unwrap()
        .allocation_bytes;
    assert_eq!(first, second);
    assert_eq!(first_bytes, second_bytes,
        "commitment must warm the packed immutable cache; a CPU-expanded commit leaves the first fold to upload its source");
}
