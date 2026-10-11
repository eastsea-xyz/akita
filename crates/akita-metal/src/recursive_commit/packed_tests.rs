use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_prover::backend::{tensor_pack_recursive_witness, RecursiveWitnessFlat};
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
use akita_types::SetupMatrixCapacity;
use jolt_field::{Ext2, Ring};

mod runtime;

fn width_witness<const D: usize>(rings: usize, width: u8, state: &mut u64) -> RecursiveWitnessFlat {
    let bound = 1i16 << (width - 1);
    RecursiveWitnessFlat::from_i8_digits(
        (0..rings * D)
            .map(|index| {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                match index % 11 {
                    0 => (-bound) as i8,
                    1 => (bound - 1) as i8,
                    _ => (((*state >> 32) as i16).rem_euclid(2 * bound) - bound) as i8,
                }
            })
            .collect(),
    )
}

fn all_widths<const D: usize>() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        18,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 4 * 3 * D,
        },
        None,
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut state = 0x467f_b9c2_0713_eda1;
    for width in 1..=8 {
        let witnesses = [
            width_witness::<D>(19, width, &mut state),
            width_witness::<D>(70, width, &mut state),
        ];
        for planes in [1, 2, 3] {
            let plan = CommitInnerPlan {
                n_a: 3,
                num_positions_per_block: 4,
                num_digits_inner: planes,
                log_basis_inner: 3,
            };
            let sources = witnesses
                .iter()
                .map(|w| w.view::<F, D>().unwrap())
                .collect::<Vec<_>>();
            let expected = cpu
                .commit_inner_group(&cpu_prepared, sources.clone(), plan)
                .unwrap();
            for _ in 0..2 {
                backend.begin_opening_metrics().unwrap();
                let actual = backend
                    .commit_inner_group(&prepared, sources.clone(), plan)
                    .unwrap();
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(
                        actual.inner_rows, expected.inner_rows,
                        "D{D}, width={width}, planes={planes}"
                    );
                }
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
fn packed_recursive_commit_random_all_widths_d64_d128() {
    all_widths::<64>();
    all_widths::<128>();
}

#[test]
fn packed_recursive_commit_random_all_widths_d256_d512() {
    all_widths::<256>();
    all_widths::<512>();
}

fn tensor_tails<const D: usize>() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        18,
        1,
        SetupMatrixCapacity {
            num_field_elements: 2 * 4 * 3 * D,
        },
        None,
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut logical_digits = (0..D + 2).map(|i| (i % 7) as i8 - 3).collect::<Vec<_>>();
    // The final extension coordinate embeds at D+D/4 and D+3D/4,
    // beyond the logical end. Its original periodic value is zero at D128.
    logical_digits[D + 1] = 1;
    let logical = RecursiveWitnessFlat::from_i8_digits(logical_digits);
    let witness = tensor_pack_recursive_witness::<F, Ext2<F>, D>(&logical).unwrap();
    let physical = witness.to_i8_digits();
    assert!(
        physical[witness.live_coeff_len()..]
            .iter()
            .any(|&value| value != 0),
        "fixture must contain a nonzero stored tensor tail"
    );
    let view = witness.view::<F, D>().unwrap();
    for planes in [1, 2, 3] {
        let plan = CommitInnerPlan {
            n_a: 2,
            num_positions_per_block: 4,
            num_digits_inner: planes,
            log_basis_inner: 3,
        };
        let expected = cpu
            .commit_inner_group(&cpu_prepared, vec![view], plan)
            .unwrap();
        let actual = backend
            .commit_inner_group(&prepared, vec![view], plan)
            .unwrap();
        assert_eq!(
            actual[0].inner_rows, expected[0].inner_rows,
            "D{D}, physical tail, planes={planes}"
        );
    }
}

#[test]
fn packed_recursive_commit_preserves_physical_tensor_tails_and_implicit_zeros() {
    tensor_tails::<64>();
    tensor_tails::<128>();
    tensor_tails::<256>();
    tensor_tails::<512>();
}

#[test]
fn packed_recursive_commit_asymmetric_basis_one_128_planes_matches_cpu() {
    const D: usize = 64;
    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: 1,
        num_digits_inner: 128,
        log_basis_inner: 1,
    };
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        18,
        1,
        SetupMatrixCapacity {
            num_field_elements: 128 * D,
        },
        None,
    )
    .unwrap();
    let witness = RecursiveWitnessFlat::from_i8_digits(
        (0..D)
            .map(|i| match i % 7 {
                0 => -128,
                1 => 127,
                2 => 1,
                3 => -1,
                4 => 0,
                5 => 2,
                _ => -2,
            })
            .collect(),
    );
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
}

#[test]
fn packed_recursive_commit_invalid_second_source_rejects_before_matrix_work() {
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
    let good = RecursiveWitnessFlat::from_i8_digits(vec![1; D]);
    let empty = RecursiveWitnessFlat::from_i8_digits(Vec::new());
    let sources = vec![good.view::<F, D>().unwrap(), empty.view::<F, D>().unwrap()];
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    backend.begin_opening_metrics().unwrap();
    assert!(backend
        .commit_inner_group(&prepared, sources, plan)
        .is_err());
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 0);
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
    assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
}

#[test]
fn packed_recursive_commit_i16_basis_preserves_require_and_prefer_policy() {
    const D: usize = 128;
    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: 2,
        num_digits_inner: 2,
        log_basis_inner: 9,
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
    let witness = RecursiveWitnessFlat::from_i8_digits((0..D).map(|i| i as i8).collect());
    let view = witness.view::<F, D>().unwrap();
    let cpu = CpuBackend::DEFAULT;
    let expected = cpu
        .commit_inner_group(&cpu.prepare_setup(&setup).unwrap(), vec![view], plan)
        .unwrap();
    for policy in [
        MetalExecutionPolicy::RequireMetal,
        MetalExecutionPolicy::PreferMetal,
    ] {
        let backend = MetalBackend::new(policy).unwrap();
        let prepared = backend.prepare_setup(&setup).unwrap();
        backend.begin_opening_metrics().unwrap();
        let actual = backend.commit_inner_group(&prepared, vec![view], plan);
        let metrics = backend.last_opening_metrics().unwrap().unwrap();
        match policy {
            MetalExecutionPolicy::RequireMetal => {
                assert!(actual.is_err());
                assert_eq!(metrics.cpu_fallback_calls, 0);
            }
            MetalExecutionPolicy::PreferMetal => {
                assert_eq!(actual.unwrap()[0].inner_rows, expected[0].inner_rows);
                assert_eq!(metrics.cpu_fallback_calls, 1);
                assert_eq!(metrics.cpu_fallback_work_units, 1);
            }
        }
        assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
    }
}
