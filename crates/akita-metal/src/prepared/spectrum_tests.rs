use super::*;
use crate::runtime::{RecursiveCommitInput, FP128_D512_LINEAR_RELATION_NUM_PRIMES};
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend, RecursiveWitnessFlat};
use akita_types::SetupMatrixCapacity;

fn params<const D: usize>(rows: usize, cols: usize, blocks: usize) -> RecursiveCommitParams {
    let group = MetalRuntime::recursive_commit_blocks_per_group::<D>();
    RecursiveCommitParams {
        num_blocks: blocks as u64,
        blocks_per_group: group as u64,
        num_block_groups: blocks.div_ceil(group) as u64,
        num_rows: rows as u64,
        num_cols: cols as u64,
        ring_d: D as u64,
        num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
        matrix_rings: (rows * cols) as u64,
        output_coefficients: (blocks * rows * D) as u64,
        rhs_abs_bound: 128,
    }
}

fn assert_linear_prefix_shared<const D: usize>() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 24 * D,
        },
        None,
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let runtime = metal.runtime().unwrap();
    let linear = prepared.linear_matrix::<D>(runtime, 24).unwrap();
    let recursive = prepared
        .recursive_matrix::<D>(runtime, params::<D>(3, 4, 2))
        .unwrap();
    assert!(
        recursive.cache_hit,
        "existing ordinary spectrum must serve recursive prefix D{D}"
    );
    assert!(Arc::ptr_eq(&linear.buffer, &recursive.buffer));
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    assert_eq!(
        prepared.matrix_cache_bytes().unwrap(),
        24 * 2 * D * 6 * size_of::<i32>()
    );
}

#[test]
fn shared_spectrum_linear_then_recursive_all_dimensions_reuses_one_buffer() {
    assert_linear_prefix_shared::<64>();
    assert_linear_prefix_shared::<128>();
    assert_linear_prefix_shared::<256>();
    assert_linear_prefix_shared::<512>();
}

fn assert_growth_and_borrowers<const D: usize>() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 24 * D,
        },
        None,
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let runtime = metal.runtime().unwrap();
    let first = prepared
        .recursive_matrix::<D>(runtime, params::<D>(3, 4, 2))
        .unwrap();
    let mut retained = vec![first];
    for (rows, cols) in [(3, 5), (1, 20), (3, 8)] {
        retained.push(
            prepared
                .recursive_matrix::<D>(runtime, params::<D>(rows, cols, 2))
                .unwrap(),
        );
    }
    assert_eq!(
        prepared.matrix_cache_entries().unwrap(),
        1,
        "one largest per D, independent of rows/columns"
    );
    assert_eq!(
        prepared.matrix_cache_bytes().unwrap(),
        24 * 2 * D * 6 * size_of::<i32>()
    );
    let reshaped = prepared
        .recursive_matrix::<D>(runtime, params::<D>(2, 7, 2))
        .unwrap();
    assert!(reshaped.cache_hit);
    assert!(Arc::ptr_eq(&reshaped.buffer, &retained[3].buffer));
    let ordinary = prepared.linear_matrix::<D>(runtime, 6).unwrap();
    assert!(ordinary.cache_hit);
    assert!(Arc::ptr_eq(&ordinary.buffer, &retained[3].buffer));
    assert!(prepared
        .recursive_matrix::<D>(runtime, params::<D>(3, 9, 2))
        .is_err());
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let shapes = [(3, 4), (3, 5), (1, 20), (3, 8)];
    let references = shapes
        .into_iter()
        .map(|(rows, cols)| {
            let digits = (0..2 * cols * D)
                .map(|i| (i % 7) as i8 - 3)
                .collect::<Vec<_>>();
            let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
            let expected = cpu
                .commit_inner_group(
                    &cpu_prepared,
                    vec![witness.view::<F, D>().unwrap()],
                    CommitInnerPlan {
                        n_a: rows,
                        num_positions_per_block: cols,
                        num_digits_inner: 1,
                        log_basis_inner: 3,
                    },
                )
                .unwrap();
            (digits, expected[0].inner_rows.coeffs().to_vec())
        })
        .collect::<Vec<_>>();
    drop(cpu_prepared);
    drop(prepared);
    drop(setup);
    // Older evicted borrowers must preserve exact products after every original
    // setup/preparation handle is gone, including each physical prime stride.
    for (index, ((rows, cols), (digits, expected))) in
        shapes.into_iter().zip(references).enumerate()
    {
        let output = runtime
            .dispatch_fp128_recursive_commit::<D>(
                &retained[index].view(),
                RecursiveCommitInput::Raw(&digits),
                params::<D>(rows, cols, 2),
            )
            .unwrap();
        let fields = output
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(i, value)| value.into_field(i).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            fields, expected,
            "evicted borrower D{D}, rows{rows}, cols{cols}"
        );
    }
}

#[test]
fn shared_spectrum_growth_replaces_cache_and_retains_evicted_raw_borrowers() {
    assert_growth_and_borrowers::<64>();
    assert_growth_and_borrowers::<128>();
    assert_growth_and_borrowers::<256>();
    assert_growth_and_borrowers::<512>();
}

fn assert_mixed_raw_and_packed<const D: usize>() {
    const MAX_RINGS: usize = 72;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: MAX_RINGS * D,
        },
        None,
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let runtime = metal.runtime().unwrap();
    let largest = prepared.linear_matrix::<D>(runtime, MAX_RINGS).unwrap();
    let mut random = 0x715acd31b907_u64;
    for (rows, positions, planes) in [(3, 8, 3), (1, 7, 1), (3, 3, 1), (2, 5, 3)] {
        let digits = (0..24 * D)
            .map(|index| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                match index % 17 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    _ => (random >> 32) as i8,
                }
            })
            .collect::<Vec<_>>();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
        let plan = CommitInnerPlan {
            n_a: rows,
            num_positions_per_block: positions,
            num_digits_inner: planes,
            log_basis_inner: 3,
        };
        let view = witness.view::<F, D>().unwrap();
        let expected = cpu
            .commit_inner_group(&cpu_prepared, vec![view], plan)
            .unwrap();
        let actual = metal
            .commit_inner_group(&prepared, vec![view], plan)
            .unwrap();
        assert_eq!(
            actual[0].inner_rows,
            expected[0].inner_rows,
            "packed prefix D{D}, rows{rows}, cols{}",
            positions * planes
        );
        if planes == 1 {
            let blocks = 24usize.div_ceil(positions);
            let p = params::<D>(rows, positions, blocks);
            let matrix = prepared.recursive_matrix::<D>(runtime, p).unwrap();
            let mut raw = digits.clone();
            raw.resize(blocks * positions * D, 0);
            let output = runtime
                .dispatch_fp128_recursive_commit::<D>(
                    &matrix.view(),
                    RecursiveCommitInput::Raw(&raw),
                    p,
                )
                .unwrap();
            let fields = output
                .coefficients
                .into_iter()
                .enumerate()
                .map(|(i, value)| value.into_field(i).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(fields, expected[0].inner_rows.coeffs());
        }
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
        assert_eq!(
            prepared.matrix_cache_bytes().unwrap(),
            MAX_RINGS * 2 * D * 6 * size_of::<i32>()
        );
    }
    assert!(Arc::ptr_eq(
        &largest.buffer,
        &prepared.linear_matrix::<D>(runtime, 1).unwrap().buffer
    ));
}

#[test]
fn shared_spectrum_mixed_raw_packed_smaller_reshapes_match_cpu_all_dimensions() {
    assert_mixed_raw_and_packed::<64>();
    assert_mixed_raw_and_packed::<128>();
    assert_mixed_raw_and_packed::<256>();
    assert_mixed_raw_and_packed::<512>();
}
