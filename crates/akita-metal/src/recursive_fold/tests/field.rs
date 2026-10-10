use super::*;
use crate::runtime::MetalRuntime;
use akita_prover::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, DecomposeParams,
};

fn params<const D: usize>(digits: usize, basis: u32) -> RecursiveFoldParams {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    RecursiveFoldParams {
        stored_coefficients: (7 * D) as u64,
        physical_coefficients: (7 * D) as u64,
        live_rings: 7,
        positions_per_block: 4,
        num_blocks: 2,
        ring_d: D as u64,
        bit_width: 0,
        num_digits: digits as u64,
        log_basis: u64::from(basis),
        threshold: Fp128Limbs::from_u128(decompose_centering_threshold(digits, basis, modulus)),
    }
}

fn field_sparse_parity<const D: usize>() {
    let runtime = MetalRuntime::new().unwrap();
    let mut state = 0x04ee_0123_aa59_841c;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    for (digits, basis) in [(128, 1), (64, 2), (32, 4), (16, 8), (9, 15), (8, 16)] {
        let p = params::<D>(digits, basis);
        let threshold = p.threshold.to_u128();
        let boundaries = [
            0,
            1,
            modulus - 1,
            threshold,
            threshold + 1,
            (1u128 << 127) - 1,
            1u128 << 127,
        ];
        let fields = (0..7 * D)
            .map(|i| {
                let value = if i % 11 < boundaries.len() {
                    boundaries[i % 11]
                } else {
                    (u128::from(next(&mut state)) << 64) | u128::from(next(&mut state))
                };
                F::from_u128_reduced(value)
            })
            .collect::<Vec<_>>();
        let buffer = runtime
            .shared_buffer_from_slice(
                &fields
                    .iter()
                    .copied()
                    .map(MetalField::into_device)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let challenges = challenges::<D>(&mut state, 2);
        let terms = challenges
            .iter()
            .enumerate()
            .flat_map(|(block, challenge)| {
                challenge.positions.iter().zip(&challenge.coeffs).map(
                    move |(&shift, &coefficient)| RecursiveSparseTerm {
                        block: block as u64,
                        shift,
                        coefficient: i32::from(coefficient),
                    },
                )
            })
            .collect::<Vec<_>>();
        let rings = fields
            .chunks_exact(D)
            .map(|row| CyclotomicRing::from_coefficients(std::array::from_fn(|i| row[i])))
            .collect::<Vec<_>>();
        let cpu_params = DecomposeParams {
            threshold,
            q: modulus,
            mask: (1i128 << basis) - 1,
            half_b: 1i128 << (basis - 1),
            b_val: 1i128 << basis,
            log_basis: basis,
            overflow_possible: modulus.saturating_sub(threshold) > i128::MAX as u128,
        };
        let expected = balanced_ring_decompose_fold_partitioned::<F, D>(
            &rings,
            &challenges,
            4,
            digits,
            &cpu_params,
        )
        .into_flattened();
        let actual = runtime
            .dispatch_recursive_sparse_fold(&[(
                RecursiveFoldSource::Field {
                    buffer: buffer.into(),
                },
                p,
                terms,
            )])
            .unwrap();
        assert_eq!(
            actual.centered_coefficients, expected,
            "D{D}, log_basis={basis}, digits={digits}"
        );
        assert!(
            actual.coefficients.is_empty(),
            "sparse path must not materialize unused field storage"
        );
    }
}

#[test]
fn recursive_balanced_sparse_random_matches_exact_cpu_partitioned_helper() {
    field_sparse_parity::<64>();
    field_sparse_parity::<128>();
    field_sparse_parity::<256>();
    field_sparse_parity::<512>();
}

#[test]
fn recursive_signed_bound_and_invalid_term_admission_precede_submission() {
    let runtime = MetalRuntime::new().unwrap();
    let p = params::<128>(8, 16);
    let invalid = [RecursiveSparseTerm {
        block: 2,
        shift: 0,
        coefficient: 1,
    }];
    assert!(runtime
        .admit_recursive_sparse_plans([(&p, &invalid[..])])
        .is_err());
    let invalid = [RecursiveSparseTerm {
        block: 0,
        shift: 128,
        coefficient: 1,
    }];
    assert!(runtime
        .admit_recursive_sparse_plans([(&p, &invalid[..])])
        .is_err());
    let too_wide = [RecursiveSparseTerm {
        block: 0,
        shift: 0,
        coefficient: i32::MIN,
    }];
    assert!(runtime
        .admit_recursive_sparse_plans([(&p, &too_wide[..])])
        .is_err());
    let too_many = [RecursiveSparseTerm {
        block: 0,
        shift: 0,
        coefficient: 65536,
    }];
    assert!(matches!(
        runtime.admit_recursive_sparse_plans([(&p, &too_many[..])]),
        Err(MetalCommitError::ShapeOverflow(_))
    ));
    let safe = [RecursiveSparseTerm {
        block: 0,
        shift: 127,
        coefficient: 65535,
    }];
    assert!(runtime
        .admit_recursive_sparse_plans([(&p, &safe[..])])
        .is_ok());
    assert!(runtime
        .admit_recursive_sparse_plans([(&p, &safe[..]), (&p, &safe[..])])
        .is_err());
}

#[test]
fn recursive_field_evaluate_fold_uses_canonical_negacyclic_arithmetic() {
    const D: usize = 128;
    let runtime = MetalRuntime::new().unwrap();
    let p = params::<D>(1, 1);
    let mut state = 0x569e_b41a_6190_763c;
    let fields = (0..7 * D)
        .map(|_| {
            F::from_u128_reduced(
                (u128::from(next(&mut state)) << 64) | u128::from(next(&mut state)),
            )
        })
        .collect::<Vec<_>>();
    let positions = (0..4)
        .map(|position| {
            let mut coeffs = [F::zero(); D];
            coeffs[0] = F::from_u64(position + 2);
            coeffs[D - 1] = F::from_i64(-(position as i64) - 3);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect::<Vec<_>>();
    let folds = (0..2)
        .map(|block| {
            let mut coeffs = [F::zero(); D];
            coeffs[D / 2] = F::from_u64(block + 5);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect::<Vec<_>>();
    let terms = |rings: &[CyclotomicRing<F, D>]| {
        rings
            .iter()
            .enumerate()
            .flat_map(|(position, ring)| {
                ring.coeffs
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(move |(shift, weight)| RecursiveWeightTerm {
                        position: position as u64,
                        shift: shift as u64,
                        weight: weight.into_device(),
                    })
            })
            .collect::<Vec<_>>()
    };
    let buffer = runtime.shared_buffer_from_slice(&fields).unwrap();
    let actual = runtime
        .dispatch_recursive_evaluate_fold(
            &RecursiveFoldSource::Field {
                buffer: buffer.into(),
            },
            p,
            &terms(&positions),
            &terms(&folds),
        )
        .unwrap();
    let rings = fields
        .chunks_exact(D)
        .map(|row| CyclotomicRing::from_coefficients(std::array::from_fn(|i| row[i])))
        .collect::<Vec<_>>();
    let expected_folded = rings
        .chunks(4)
        .map(|block| {
            block
                .iter()
                .zip(&positions)
                .fold(CyclotomicRing::zero(), |acc, (a, b)| acc + *a * *b)
        })
        .collect::<Vec<_>>();
    let expected_eval = expected_folded
        .iter()
        .zip(&folds)
        .fold(CyclotomicRing::zero(), |acc, (a, b)| acc + *a * *b);
    let expected = expected_folded
        .into_iter()
        .chain([expected_eval])
        .flat_map(|r| r.coeffs)
        .collect::<Vec<_>>();
    let actual = actual
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(i, c)| c.into_field(i).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn recursive_packed_tensor_tail_uses_separate_evaluation_and_decomposition_extents() {
    const D: usize = 128;
    let runtime = MetalRuntime::new().unwrap();
    let digits = (0..8 * D).map(|i| (i % 7) as i8 - 3).collect::<Vec<_>>();
    let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
    let (bytes, width, stored) = witness.packed_digit_storage();
    let source = RecursiveFoldSource::Packed {
        owner: Arc::from(bytes),
    };
    let p = RecursiveFoldParams {
        stored_coefficients: stored as u64,
        physical_coefficients: (8 * D) as u64,
        live_rings: 7,
        bit_width: u64::from(width),
        ..params::<D>(1, 1)
    };
    let positions = (0..4)
        .map(|i| RecursiveWeightTerm {
            weight: Fp128Limbs::from_u128(i as u128 + 2),
            position: i,
            shift: 0,
        })
        .collect::<Vec<_>>();
    let folds = (0..2)
        .map(|i| RecursiveWeightTerm {
            weight: Fp128Limbs::from_u128(i as u128 + 7),
            position: i,
            shift: 0,
        })
        .collect::<Vec<_>>();
    let first = runtime
        .dispatch_recursive_evaluate_fold(&source, p, &positions, &folds)
        .unwrap();
    let second = runtime
        .dispatch_recursive_evaluate_fold(&source, p, &positions, &folds)
        .unwrap();
    assert_eq!(first.coefficients, second.coefficients);
    assert_eq!(
        first.allocation_bytes - second.allocation_bytes,
        bytes.len()
    );
    for coefficient in 0..D {
        let last = (4..8).fold(F::zero(), |acc, ring| {
            acc + F::from_i8(digits[ring * D + coefficient]) * F::from_u64((ring - 4 + 2) as u64)
        });
        assert_eq!(
            first.coefficients[D + coefficient]
                .into_field(coefficient)
                .unwrap(),
            last,
            "evaluation must include physical tensor tail ring"
        );
    }
    let challenge = SparseChallenge {
        positions: [0].into_iter().collect(),
        coeffs: [1].into_iter().collect(),
    };
    let terms = vec![
        RecursiveSparseTerm {
            block: 0,
            shift: 0,
            coefficient: 1,
        },
        RecursiveSparseTerm {
            block: 1,
            shift: 0,
            coefficient: 1,
        },
    ];
    let actual = runtime
        .dispatch_recursive_sparse_fold(&[(source, p, terms)])
        .unwrap();
    let planes = digits[..7 * D]
        .chunks_exact(D)
        .map(|row| std::array::from_fn(|i| row[i]))
        .collect::<Vec<[i8; D]>>();
    let expected =
        akita_prover::backend::poly_helpers::cached_digit_decompose_fold_partitioned::<F, D>(
            &planes,
            &[challenge.clone(), challenge],
            4,
            1,
            u32::from(width),
        )
        .into_flattened();
    assert_eq!(
        actual.centered_coefficients, expected,
        "sparse fold must omit non-live tensor tail ring"
    );
}
