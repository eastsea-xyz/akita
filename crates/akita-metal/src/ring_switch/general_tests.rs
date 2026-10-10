use super::*;
use crate::runtime::{MetalRuntime, RelationRowsInput, RelationRowsMode, RelationRowsParams};
use crate::MetalExecutionPolicy;
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
use akita_types::SetupMatrixCapacity;

fn assert_general_parity<const D: usize>() {
    const MAX_ROWS: usize = 3;
    const MAX_COLUMNS: usize = 11;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: MAX_ROWS * MAX_COLUMNS * D,
        },
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let mut state = 0x407324913fe2_u64;
    for rows in [1, 3] {
        for (d_cols, b_cols, a_cols) in [(1, 3, 7), (7, 11, 3)] {
            let mut digit_rows = |columns| {
                (0..columns)
                    .map(|_| {
                        std::array::from_fn(|_| {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            (state >> 32) as i8
                        })
                    })
                    .collect::<Vec<[i8; D]>>()
            };
            let e_hat = digit_rows(d_cols);
            let t_hat = digit_rows(b_cols);
            let z_segment = (0..a_cols)
                .map(|column| {
                    std::array::from_fn(|coefficient| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        match (column * D + coefficient) % 17 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            _ => (state >> 32) as i32,
                        }
                    })
                })
                .collect::<Vec<[i32; D]>>();
            // A claimed norm smaller than the input must never weaken admission.
            let source = RingSwitchRelationView {
                e_hat: &e_hat,
                t_hat: &t_hat,
                z_segment: &z_segment,
                z_folded_centered_inf_norm: 13,
            };
            for active in [0, 1, 2] {
                let plan = RingSwitchRelationPlan {
                    n_d: if active == 2 { rows } else { 0 },
                    n_b: if active != 1 { rows } else { 0 },
                    n_a: if active != 0 { rows } else { 0 },
                    log_basis_open: 8,
                    log_basis_outer: 8,
                };
                let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();
                for _ in 0..2 {
                    metal.begin_opening_metrics().unwrap();
                    let actual = metal.relation_rows(&prepared, source, plan).unwrap();
                    assert_eq!(
                        actual, expected,
                        "D={D}, rows={rows}, widths={d_cols}/{b_cols}/{a_cols}, active={active}"
                    );
                    let metrics = metal.last_opening_metrics().unwrap().unwrap();
                    assert_eq!(
                        metrics.cpu_fallback_calls, 0,
                        "general relation rows must dispatch"
                    );
                    assert!(metrics.command_wall_time > std::time::Duration::ZERO);
                    assert!(prepared.matrix_cache_entries().unwrap() > 0);
                }
            }
        }
    }
}

#[test]
fn general_relation_rows_random_all_dimensions_match_cpu_and_dispatch() {
    assert_general_parity::<64>();
    assert_general_parity::<128>();
    assert_general_parity::<256>();
    assert_general_parity::<512>();
}

#[test]
fn general_relation_rows_reuse_larger_linear_prefix_with_unequal_widths() {
    const D: usize = 128;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 11 * D,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let runtime = metal.runtime().unwrap();
    let large = prepared.linear_matrix::<D>(runtime, 33).unwrap();
    let small = prepared.linear_matrix::<D>(runtime, 7).unwrap();
    assert!(small.cache_hit);
    assert_eq!(
        std::sync::Arc::as_ptr(&large.buffer),
        std::sync::Arc::as_ptr(&small.buffer)
    );
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    let t_hat = vec![[-128i8; D]; 7];
    let z_segment = vec![[i32::MIN; D]; 3];
    let source = RingSwitchRelationView {
        e_hat: &[],
        t_hat: &t_hat,
        z_segment: &z_segment,
        z_folded_centered_inf_norm: 0,
    };
    let plan = RingSwitchRelationPlan {
        n_d: 0,
        n_b: 3,
        n_a: 1,
        log_basis_open: 3,
        log_basis_outer: 8,
    };
    let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();
    metal.begin_opening_metrics().unwrap();
    let actual = metal.relation_rows(&prepared, source, plan).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    assert_eq!(
        metal
            .last_opening_metrics()
            .unwrap()
            .unwrap()
            .cpu_fallback_calls,
        0
    );
}

#[test]
fn general_relation_rows_reject_malformed_before_any_matrix_transform() {
    const D: usize = 128;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: D,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let invalid = [[4i8; D]];
    let source = RingSwitchRelationView {
        e_hat: &[],
        t_hat: &invalid,
        z_segment: &[],
        z_folded_centered_inf_norm: 0,
    };
    let plan = RingSwitchRelationPlan {
        n_d: 0,
        n_b: 1,
        n_a: 0,
        log_basis_open: 3,
        log_basis_outer: 3,
    };
    assert!(metal.relation_rows(&prepared, source, plan).is_err());
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 0);
    let source = RingSwitchRelationView {
        t_hat: &[],
        ..source
    };
    assert!(metal.relation_rows(&prepared, source, plan).is_err());
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 0);
    let t_hat = [[1i8; D]; 2];
    let source = RingSwitchRelationView {
        t_hat: &t_hat,
        ..source
    };
    assert!(metal.relation_rows(&prepared, source, plan).is_err());
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 0);
}

#[test]
fn general_relation_rows_runtime_checks_actual_bound_shape_and_buffers() {
    const D: usize = 128;
    let runtime = MetalRuntime::new().unwrap();
    let data = [[i32::MIN; D]];
    let input = RelationRowsInput::from_i32(&data, 0);
    let valid = RelationRowsParams::new(1, 1, RelationRowsMode::High, &input).unwrap();
    assert_eq!(valid.rhs_abs_bound, 1u64 << 31);
    assert!(runtime.supports_fp128_relation_rows::<D>(valid));
    let matrix = runtime.shared_buffer_from_slice(&[0i32]).unwrap();
    for params in [
        valid,
        RelationRowsParams { mode: 3, ..valid },
        RelationRowsParams {
            rhs_abs_bound: 1,
            ..valid
        },
        RelationRowsParams {
            output_coefficients: 1,
            ..valid
        },
        RelationRowsParams {
            ntt_size: 1024,
            ..valid
        },
        RelationRowsParams {
            num_cols: 2,
            ..valid
        },
        RelationRowsParams {
            matrix_rings: u64::MAX,
            ..valid
        },
    ] {
        assert!(runtime
            .dispatch_fp128_relation_rows(&matrix, &input, params)
            .is_err());
    }
    assert!(
        !runtime.supports_fp128_relation_rows::<D>(RelationRowsParams {
            rhs_abs_bound: u64::MAX,
            num_cols: 4096,
            matrix_rings: 4096,
            ..valid
        })
    );
    let adequate = runtime
        .shared_buffer_from_slice(&vec![0i32; 2 * D * 6])
        .unwrap();
    assert!(runtime
        .dispatch_fp128_relation_rows(
            &adequate,
            &input,
            RelationRowsParams {
                rhs_abs_bound: 1,
                ..valid
            }
        )
        .is_err());
    assert!(RelationRowsInput::from_i8(&[[4i8; D]], 3).is_err());
    assert!(RelationRowsInput::from_i8(&[[0i8; D]], 0).is_err());
}
