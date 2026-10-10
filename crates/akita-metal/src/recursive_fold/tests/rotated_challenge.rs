use super::*;
use crate::runtime::MetalRuntime;
use akita_prover::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, DecomposeParams,
};

fn repeated(entries: usize, last: i8) -> SparseChallenge {
    let mut coefficients = vec![1; entries];
    *coefficients.last_mut().unwrap() = last;
    SparseChallenge {
        positions: std::iter::repeat_n(0, entries).collect(),
        coeffs: coefficients.into_iter().collect(),
    }
}

fn packed_duplicate_parity<const D: usize>() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; D]);
    let view = witness.view::<F, D>().unwrap();
    for entries in [41, 42, 43] {
        for last in [1, -3, 0, i8::MIN] {
            let challenges = [repeated(entries, last)];
            let plan = DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 0,
            };
            let expected =
                OpeningFoldKernel::decompose_fold(&CpuBackend::DEFAULT, None, view, plan).unwrap();
            backend.begin_opening_metrics().unwrap();
            let actual = OpeningFoldKernel::decompose_fold(&backend, None, view, plan).unwrap();
            assert_eq!(
                actual, expected,
                "D{D}, {entries} repeated terms, last={last}"
            );
            let coefficient = if D == 64 && entries >= 42 {
                i32::from(last)
            } else {
                entries as i32 - 1 + i32::from(last)
            };
            assert_eq!(actual.centered_coeffs_flat(), vec![coefficient; D]);
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

#[test]
fn recursive_d64_rotated_duplicates_match_cpu_at_41_42_threshold() {
    packed_duplicate_parity::<64>();
}

#[test]
fn recursive_d128_duplicates_accumulate_above_d64_rotated_threshold() {
    packed_duplicate_parity::<128>();
}

#[test]
fn recursive_d64_live_field_multi_plane_rotated_duplicates_match_cpu() {
    const D: usize = 64;
    let runtime = MetalRuntime::new().unwrap();
    let fields = (0..D)
        .map(|i| F::from_i64((i % 37) as i64 - 18))
        .collect::<Vec<_>>();
    let rings = [CyclotomicRing::from_coefficients(std::array::from_fn(
        |i| fields[i],
    ))];
    let buffer = runtime.shared_buffer_from_slice(&fields).unwrap();
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let threshold = decompose_centering_threshold(2, 4, modulus);
    let cpu = DecomposeParams {
        threshold,
        q: modulus,
        mask: 15,
        half_b: 8,
        b_val: 16,
        log_basis: 4,
        overflow_possible: modulus - threshold > i128::MAX as u128,
    };
    for entries in [41, 42, 43] {
        for last in [-3, 0, 1] {
            let challenge = repeated(entries, last);
            let mut p = RecursiveFoldParams {
                stored_coefficients: D as u64,
                physical_coefficients: D as u64,
                live_rings: 1,
                positions_per_block: 1,
                num_blocks: 1,
                ring_d: D as u64,
                bit_width: 0,
                num_digits: 2,
                log_basis: 4,
                threshold: Fp128Limbs::from_u128(threshold),
            };
            let terms = sparse_terms::<D>(
                &mut p,
                DecomposeFoldPlan {
                    challenges: std::slice::from_ref(&challenge),
                    num_positions_per_block: 1,
                    num_digits: 2,
                    log_basis: 4,
                },
            )
            .unwrap();
            assert_eq!(
                terms.len(),
                if entries < 42 {
                    entries
                } else {
                    usize::from(last != 0)
                }
            );
            let expected =
                balanced_ring_decompose_fold_partitioned::<F, D>(&rings, &[challenge], 1, 2, &cpu)
                    .into_flattened();
            let actual = runtime
                .dispatch_recursive_sparse_fold(&[(
                    RecursiveFoldSource::Field {
                        buffer: buffer.clone(),
                    },
                    p,
                    terms,
                )])
                .unwrap();
            assert_eq!(
                actual.centered_coefficients, expected,
                "live field, {entries} entries, last={last}"
            );
        }
    }
}
