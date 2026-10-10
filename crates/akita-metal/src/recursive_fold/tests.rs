use super::*;
use akita_challenges::SparseChallenge;
use akita_prover::backend::{RecursiveFoldSource as ProverSource, RecursiveWitnessFlat};
use akita_prover::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource,
};
use akita_types::{prepare_opening_point, BasisMode};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Ring};

mod field;
mod rotated_challenge;
mod setup_prefix;

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn challenges<const D: usize>(state: &mut u64, blocks: usize) -> Vec<SparseChallenge> {
    (0..blocks)
        .map(|block| SparseChallenge {
            positions: [0, (D - 1) as u32, (next(state) as usize % D) as u32, 0]
                .into_iter()
                .collect(),
            coeffs: [-128, 127, block as i8 - 3, 1].into_iter().collect(),
        })
        .collect()
}

fn assert_packed_parity<const D: usize>() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut state = 0x715c_09e4_8712_b5f3;
    for width in 1..=8 {
        for rings in [3usize, 7, 8] {
            let bound = 1i16 << (width - 1);
            let make = |state: &mut u64| {
                RecursiveWitnessFlat::from_i8_digits(
                    (0..rings * D)
                        .map(|i| match i % 19 {
                            0 => -bound as i8,
                            1 => (bound - 1) as i8,
                            _ => ((next(state) % (2 * bound) as u64) as i16 - bound) as i8,
                        })
                        .collect(),
                )
            };
            let first = ProverSource::Witness(Arc::new(make(&mut state)));
            let second = ProverSource::Witness(Arc::new(make(&mut state)));
            let sources = [&second, &first, &second]; // selected/repeated point-index order
            let positions = (0..4)
                .map(|_| {
                    F::from_u128_reduced(
                        (u128::from(next(&mut state)) << 64) | u128::from(next(&mut state)),
                    )
                })
                .collect::<Vec<_>>();
            let folds = (0..rings.div_ceil(4))
                .map(|_| F::from_u64(next(&mut state)))
                .collect::<Vec<_>>();
            let plan = OpeningFoldPlan::Base {
                live_block_weights: &folds,
                position_weights: &positions,
                num_positions_per_block: 4,
            };
            let view = <ProverSource<F> as RootOpeningSource<F, D>>::opening_view(&first).unwrap();
            let expected =
                OpeningFoldKernel::evaluate_and_fold(&CpuBackend::DEFAULT, None, view, plan)
                    .unwrap();
            for _ in 0..2 {
                backend.begin_opening_metrics().unwrap();
                assert_eq!(
                    OpeningFoldKernel::evaluate_and_fold(&backend, None, view, plan).unwrap(),
                    expected
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
            let all_challenges = challenges::<D>(&mut state, folds.len() * sources.len());
            let mut expected = vec![0i32; 4 * D];
            for (poly, challenges) in sources.iter().zip(all_challenges.chunks_exact(folds.len())) {
                let view =
                    <ProverSource<F> as RootOpeningSource<F, D>>::opening_view(poly).unwrap();
                let single = OpeningFoldKernel::decompose_fold(
                    &CpuBackend::DEFAULT,
                    None,
                    view,
                    DecomposeFoldPlan {
                        challenges,
                        num_positions_per_block: 4,
                        num_digits: 1,
                        log_basis: 0,
                    },
                )
                .unwrap();
                for (dst, &value) in expected.iter_mut().zip(single.centered_coeffs_flat()) {
                    *dst += value;
                }
            }
            let batch =
                <ProverSource<F> as RootOpeningSource<F, D>>::opening_batch(&sources).unwrap();
            for _ in 0..2 {
                backend.begin_opening_metrics().unwrap();
                let actual = OpeningBatchKernel::decompose_fold_batch(
                    &backend,
                    None,
                    batch,
                    DecomposeFoldBatchPlan::Sparse {
                        challenges: &all_challenges,
                        num_positions_per_block: 4,
                        num_digits: 1,
                        log_basis: 0,
                    },
                )
                .unwrap();
                let BatchDecomposeFoldOutcome::Fused(actual) = actual else {
                    panic!("expected fused Metal batch");
                };
                assert_eq!(actual.centered_coeffs_flat(), expected);
                assert_eq!(
                    backend
                        .last_opening_metrics()
                        .unwrap()
                        .unwrap()
                        .cpu_fallback_calls,
                    0
                );
            }
        }
    }
}

#[test]
fn recursive_packed_random_cpu_metal_parity_all_ring_dimensions() {
    assert_packed_parity::<64>();
    assert_packed_parity::<128>();
    assert_packed_parity::<256>();
    assert_packed_parity::<512>();
}

#[test]
fn recursive_subfield_fold_preserves_sparse_shift_and_physical_order() {
    const D: usize = 128;
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let witness =
        RecursiveWitnessFlat::from_i8_digits((0..7 * D).map(|i| (i % 11) as i8 - 5).collect());
    let view = witness.view::<F, D>().unwrap();
    let alpha_bits = D.trailing_zeros() as usize;
    let mut point = vec![Ext2::<F>::zero(); alpha_bits + 3];
    for (i, coordinate) in point.iter_mut().enumerate().skip(alpha_bits) {
        *coordinate =
            Ext2::from_base_slice(&[F::from_u64((i + 2) as u64), F::from_u64((i + 7) as u64)]);
    }
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let prepared =
            prepare_opening_point::<F, Ext2<F>, D>(&point, basis, 4, 2, alpha_bits).unwrap();
        let multipliers = prepared.ring_multiplier_point.as_subfield().unwrap();
        let plan = OpeningFoldPlan::Subfield {
            multipliers,
            num_positions_per_block: 4,
        };
        let expected =
            OpeningFoldKernel::evaluate_and_fold(&CpuBackend::DEFAULT, None, view, plan).unwrap();
        backend.begin_opening_metrics().unwrap();
        assert_eq!(
            OpeningFoldKernel::evaluate_and_fold(&backend, None, view, plan).unwrap(),
            expected
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
}
