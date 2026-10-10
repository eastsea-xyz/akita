use super::*;
use crate::field::Fp128Limbs;
use crate::runtime::{
    MetalRuntime, PackedRecursiveCommitParams, RecursiveCommitInput, RecursiveCommitParams,
    FP128_D512_LINEAR_RELATION_NUM_PRIMES,
};
use akita_algebra::{
    balanced_decompose_coefficients_pow2_i8_into,
    ring::cyclotomic::{decompose_centering_threshold, BalancedDecomposePow2Params},
};
use jolt_field::{One, Zero};
use std::sync::Arc;

fn params<const D: usize>(planes: usize) -> RecursiveCommitParams {
    let grouped = MetalRuntime::recursive_commit_blocks_per_group::<D>();
    RecursiveCommitParams {
        num_blocks: 5,
        blocks_per_group: grouped as u64,
        num_block_groups: 5usize.div_ceil(grouped) as u64,
        num_rows: 1,
        num_cols: (4 * planes) as u64,
        ring_d: D as u64,
        num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
        matrix_rings: (4 * planes) as u64,
        output_coefficients: (5 * D) as u64,
        rhs_abs_bound: if planes == 1 { 128 } else { 4 },
    }
}

fn raw_packed_parity<const D: usize>() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        18,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 3 * D,
        },
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let runtime = backend.runtime().unwrap();
    let full = width_witness::<D>(20, 7, &mut 0x825a_051f_139b_4061).to_i8_digits();
    let witness = RecursiveWitnessFlat::from_i8_digits(full[..19 * D + 3].to_vec());
    let (bytes, width, stored) = witness.packed_digit_storage();
    let owner: Arc<[u8]> = Arc::from(bytes);
    let physical = 20 * D; // partial final stored ring with an implicit zero suffix
    let decoded = witness.to_i8_digits();
    let modulus = (-F::one()).to_canonical_u128() + 1;
    for planes in [1, 2, 3] {
        let p = params::<D>(planes);
        // CPU expansion is the test oracle only; production has no expanded input.
        let mut raw = vec![0i8; p.num_blocks as usize * p.num_cols as usize * D];
        if planes == 1 {
            raw[..decoded.len()].copy_from_slice(&decoded);
        } else {
            let decomposition = BalancedDecomposePow2Params::new(planes, 3, modulus);
            for (ring, output) in decoded.chunks(D).zip(raw.chunks_exact_mut(planes * D)) {
                let mut coefficients = vec![F::zero(); D];
                for (coefficient, &digit) in coefficients.iter_mut().zip(ring) {
                    *coefficient = F::from_i8(digit);
                }
                balanced_decompose_coefficients_pow2_i8_into(&coefficients, output, &decomposition);
            }
        }
        let matrix = prepared.recursive_matrix::<D>(runtime, p).unwrap();
        let expected = runtime
            .dispatch_fp128_recursive_commit::<D>(
                &matrix.buffer,
                RecursiveCommitInput::Raw(&raw),
                p,
            )
            .unwrap();
        let actual = runtime
            .dispatch_fp128_recursive_commit::<D>(
                &matrix.buffer,
                RecursiveCommitInput::Packed {
                    owner: owner.clone(),
                    params: PackedRecursiveCommitParams {
                        stored_coefficients: stored as u64,
                        physical_coefficients: physical as u64,
                        ring_d: D as u64,
                        bit_width: u64::from(width),
                        num_digits: planes as u64,
                        log_basis: 3,
                        threshold: Fp128Limbs::from_u128(decompose_centering_threshold(
                            planes, 3, modulus,
                        )),
                    },
                },
                p,
            )
            .unwrap();
        assert_eq!(
            actual.coefficients, expected.coefficients,
            "D{D}, raw reference, planes={planes}"
        );
    }
}

#[test]
fn packed_recursive_commit_raw_reference_abi_matches_packed_all_dimensions() {
    raw_packed_parity::<64>();
    raw_packed_parity::<128>();
    raw_packed_parity::<256>();
    raw_packed_parity::<512>();
}

#[test]
fn packed_recursive_commit_source_descriptor_rejects_before_upload() {
    let runtime = MetalRuntime::new().unwrap();
    let p = params::<128>(1);
    let valid = PackedRecursiveCommitParams {
        stored_coefficients: 128,
        physical_coefficients: 128,
        ring_d: 128,
        bit_width: 3,
        num_digits: 1,
        log_basis: 3,
        threshold: Fp128Limbs::default(),
    };
    for bad in [
        PackedRecursiveCommitParams {
            bit_width: 0,
            ..valid
        },
        PackedRecursiveCommitParams {
            bit_width: 9,
            ..valid
        },
        PackedRecursiveCommitParams {
            num_digits: 0,
            ..valid
        },
        PackedRecursiveCommitParams {
            physical_coefficients: 129,
            ..valid
        },
        PackedRecursiveCommitParams {
            stored_coefficients: 129,
            ..valid
        },
    ] {
        let input = RecursiveCommitInput::Packed {
            owner: Arc::from(vec![0; 48]),
            params: bad,
        };
        assert!(runtime
            .validate_recursive_commit_input::<128>(&input, p)
            .is_err());
    }
    let short = RecursiveCommitInput::Packed {
        owner: Arc::from(vec![0; 47]),
        params: valid,
    };
    assert!(runtime
        .validate_recursive_commit_input::<128>(&short, p)
        .is_err());
    assert!(runtime
        .validate_recursive_commit_input::<128>(&RecursiveCommitInput::Raw(&[0]), p)
        .is_err());
}
