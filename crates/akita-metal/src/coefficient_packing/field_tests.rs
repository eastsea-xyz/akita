use super::*;
use crate::runtime::{FieldCoefficientPackingParams, FieldPackingSource, MetalRuntime};
use crate::MetalExecutionPolicy;
use akita_prover::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use akita_prover::compute::{RootOpeningSource, SubringCoefficientPackingBatchKernel};
use akita_types::{BasisMode, FpExtEncoding, SubringCoefficientPackingGeometry};
use jolt_field::{Ext2, ExtField, MulBaseUnreduced, Ring, Zero};

fn encode_digits(digits: &[i8], width: u8) -> Vec<u8> {
    let mut bytes = vec![0; (digits.len() * usize::from(width)).div_ceil(8)];
    for (index, &digit) in digits.iter().enumerate() {
        let raw = (digit as u8) & ((1u16 << width) - 1) as u8;
        let bit = index * usize::from(width);
        bytes[bit / 8] |= raw << (bit % 8);
        if bit % 8 + usize::from(width) > 8 {
            bytes[bit / 8 + 1] |= raw >> (8 - bit % 8);
        }
    }
    bytes
}

#[test]
fn field_packing_resident_batch_matches_component_major_reference() {
    let runtime = MetalRuntime::new().unwrap();
    let params = FieldCoefficientPackingParams {
        live_coefficients: 0,
        num_live_positions: 3,
        positions_per_block: 2,
        num_blocks: 2,
        ring_d: 128,
        stride: 2,
        subring_dimension: 64,
        extension_degree: 2,
        output_coefficients: 256,
        bit_width: 0,
    };
    let fields = (0..384)
        .map(|index| F::from_i64(index as i64 - 192))
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
    let weights = [2, 3, 5, 7, 11, 13, 17, 19].map(Fp128Limbs::from_u128);
    for width in 1..=8 {
        let bound = 1i16 << (width - 1);
        let digits = (0..381)
            .map(|index| ((index as i16 % (2 * bound)) - bound) as i8)
            .collect::<Vec<_>>();
        let bytes = encode_digits(&digits, width);
        let sources = [
            FieldPackingSource::Field {
                buffer: buffer.clone(),
                coefficients: fields.len(),
            },
            FieldPackingSource::Packed {
                bytes: &bytes,
                bit_width: width,
                stored_coefficients: digits.len(),
                live_coefficients: digits.len(),
            },
        ];
        let outcome = runtime
            .dispatch_fp128_field_coefficient_packing(&sources, &weights, params)
            .unwrap();
        let actual = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| value.into_field(index).unwrap())
            .collect::<Vec<_>>();
        let mut expected = vec![F::zero(); 512];
        for (source_index, coefficients) in [
            fields.clone(),
            digits.iter().map(|&digit| F::from_i8(digit)).collect(),
        ]
        .iter()
        .enumerate()
        {
            for (flat, &coefficient) in coefficients.iter().enumerate() {
                let position = flat / 128;
                let block = position / 2;
                let term = (position % 2) * 2 + flat % 2;
                let subring = (flat % 128) / 2;
                for coordinate in 0..2 {
                    expected[source_index * 256 + block * 128 + coordinate * 64 + subring] +=
                        coefficient * weights[coordinate * 4 + term].into_field(0).unwrap();
                }
            }
        }
        assert_eq!(actual, expected, "packed bit width {width}");
        assert!(outcome.timings.command_wall > std::time::Duration::ZERO);
    }
}

#[test]
fn field_packing_rejects_invalid_batch_before_dispatch() {
    let runtime = MetalRuntime::new().unwrap();
    let params = FieldCoefficientPackingParams {
        live_coefficients: 0,
        num_live_positions: 1,
        positions_per_block: 2,
        num_blocks: 1,
        ring_d: 128,
        stride: 2,
        subring_dimension: 64,
        extension_degree: 2,
        output_coefficients: 128,
        bit_width: 0,
    };
    let bytes = vec![0; 48];
    let source = |width, stored, live| FieldPackingSource::Packed {
        bytes: &bytes,
        bit_width: width,
        stored_coefficients: stored,
        live_coefficients: live,
    };
    let weights = vec![Fp128Limbs::from_u128(1); 8];
    for invalid in [
        source(0, 128, 128),
        source(9, 128, 128),
        source(3, 128, 129),
        source(3, 128, 0),
        source(3, 120, 120),
    ] {
        assert!(runtime
            .dispatch_fp128_field_coefficient_packing(&[invalid], &weights, params)
            .is_err());
    }
    let invalid = FieldCoefficientPackingParams {
        output_coefficients: 127,
        ..params
    };
    assert!(runtime
        .dispatch_fp128_field_coefficient_packing(&[source(3, 128, 128)], &weights, invalid)
        .is_err());
    let invalid = FieldCoefficientPackingParams {
        extension_degree: 4,
        ..params
    };
    assert!(runtime
        .dispatch_fp128_field_coefficient_packing(&[source(3, 128, 128)], &weights, invalid)
        .is_err());
    assert!(runtime
        .dispatch_fp128_field_coefficient_packing(&[source(3, 128, 128)], &weights[..7], params)
        .is_err());
    let undersized = runtime
        .shared_buffer_from_slice(&[Fp128Limbs::default()])
        .unwrap();
    assert!(runtime
        .dispatch_fp128_field_coefficient_packing(
            &[FieldPackingSource::Field {
                buffer: undersized,
                coefficients: 128
            }],
            &weights,
            params
        )
        .is_err());
}

fn assert_recursive_parity<E, const D: usize>()
where
    E: ExtField<F> + FpExtEncoding<F> + MulBaseUnreduced<F> + Ring,
{
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut state = 0x74a8_097e_59ad_213bu64;
    for live_positions in [3usize, 4, 7, 8] {
        let mut make_witness = || {
            RecursiveWitnessFlat::from_i8_digits(
                (0..live_positions * D)
                    .map(|index| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        match index % 13 {
                            0 => i8::MIN,
                            1 => i8::MAX,
                            _ => (state >> 32) as i8,
                        }
                    })
                    .collect(),
            )
        };
        let first = RecursiveFoldSource::Witness(std::sync::Arc::new(make_witness()));
        let second = RecursiveFoldSource::Witness(std::sync::Arc::new(make_witness()));
        let sources = [&first, &second];
        let num_vars = (live_positions.next_power_of_two() * D).trailing_zeros() as usize;
        let public_point = (0..num_vars)
            .map(|index| {
                E::from_base_slice(
                    &(0..E::DEGREE)
                        .map(|coordinate| F::from_u64((index * 17 + coordinate * 7 + 3) as u64))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let geometry = SubringCoefficientPackingGeometry::try_new(E::DEGREE, D, 64).unwrap();
        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let point = PreparedSubringCoefficientPackingPoint::new(
                geometry,
                basis,
                live_positions,
                2,
                num_vars,
                &public_point,
            )
            .unwrap();
            let batch =
                <RecursiveFoldSource<F> as RootOpeningSource<F, D>>::opening_batch(&sources)
                    .unwrap();
            let expected = akita_prover::CpuBackend::DEFAULT
                .coefficient_packing_partials_batch(
                    None,
                    batch,
                    SubringCoefficientPackingPlan { point: &point },
                )
                .unwrap();
            for _ in 0..2 {
                backend.begin_opening_metrics().unwrap();
                let actual = backend
                    .coefficient_packing_partials_batch(
                        None,
                        batch,
                        SubringCoefficientPackingPlan { point: &point },
                    )
                    .unwrap();
                assert_eq!(
                    actual,
                    expected,
                    "D={D}, degree={}, live={live_positions}, basis={basis:?}",
                    E::DEGREE
                );
                let metrics = backend.last_opening_metrics().unwrap().unwrap();
                assert_eq!(metrics.cpu_fallback_calls, 0);
                assert!(metrics.command_wall_time > std::time::Duration::ZERO);
            }
        }
    }
}

#[test]
fn recursive_coefficient_packing_degree_two_random_matches_cpu_and_dispatches() {
    assert_recursive_parity::<F, 64>();
    assert_recursive_parity::<F, 128>();
    assert_recursive_parity::<Ext2<F>, 128>();
    assert_recursive_parity::<Ext2<F>, 512>();
}

#[test]
fn recursive_coefficient_packing_mixed_setup_prefix_degree_two_reuses_resident_matrix() {
    use akita_types::{
        sample_akita_setup_seed, AkitaCommitmentHint, AkitaExpandedSetup, AkitaSetupDescriptor,
        BasisMode, CommittedGroupParams, FlatMatrix, GroupCommitPhaseParams,
        InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
        PreparedSubringCoefficientPackingPoint, SetupPrefixPublicCommitment, SetupPrefixSlot,
        SetupPrefixSlotId, SisModulusProfileId, SubringCoefficientPackingGeometry,
    };

    const D: usize = 128;
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(D).unwrap(),
    )
    .with_decomp(4, 4, 2, 2, 2)
    .unwrap();
    let inner = &params.inner().matrix;
    let inner_bound =
        *akita_types::sis::inner_coeff_linf_bounds(inner.sis_modulus_profile(), D as u32)
            .first()
            .expect("audited setup-prefix A bound");
    params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        D,
    );
    let outer = &params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        D,
    );
    let profile =
        GroupCommitPhaseParams::try_from_params(PolynomialGroupLayout::singleton(9), &params)
            .expect("valid setup-prefix profile");
    let fields = (0..512)
        .map(|index| F::from_i64((index % 17) as i64 - 8))
        .collect::<Vec<_>>();
    let expanded = std::sync::Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 9,
                max_num_batched_polys: 1,
                num_field_elements: fields.len(),
                setup_seed: sample_akita_setup_seed(),
            },
            FlatMatrix::from_flat_data(fields.clone()),
        ),
    );
    let slot = std::sync::Arc::new(SetupPrefixSlot {
        id: SetupPrefixSlotId {
            natural_len: 400,
            commitment_profile: profile,
        },
        commitment: SetupPrefixPublicCommitment { rows: Vec::new() },
        hint: AkitaCommitmentHint::new(1, Vec::new()).unwrap(),
    });
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let setup = akita_prover::AkitaProverSetup {
        expanded: expanded.clone(),
        prefix_slots: akita_types::SetupPrefixProverRegistry::new(
            expanded.descriptor().setup_seed.clone(),
        ),
    };
    use akita_prover::ComputeBackendSetup;
    let prepared = backend.prepare_setup(&setup).unwrap();
    let first = RecursiveFoldSource::Witness(std::sync::Arc::new(
        RecursiveWitnessFlat::from_i8_digits((0..512).map(|index| (index % 8) as i8 - 4).collect()),
    ));
    let prefix = RecursiveFoldSource::SetupPrefix { expanded, slot };
    let second = RecursiveFoldSource::Witness(std::sync::Arc::new(
        RecursiveWitnessFlat::from_i8_digits((0..512).map(|index| index as i8).collect()),
    ));
    let sources = [&first, &prefix, &second, &prefix];
    let batch =
        <RecursiveFoldSource<F> as RootOpeningSource<F, D>>::opening_batch(&sources).unwrap();
    type E = Ext2<F>;
    let public_point = (0..9)
        .map(|index| {
            E::from_base_slice(&[
                F::from_u64((index + 2) as u64),
                F::from_u64((2 * index + 3) as u64),
            ])
        })
        .collect::<Vec<_>>();
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let geometry = SubringCoefficientPackingGeometry::try_new(2, D, 64).unwrap();
        let point =
            PreparedSubringCoefficientPackingPoint::new(geometry, basis, 4, 4, 9, &public_point)
                .unwrap();
        let expected = akita_prover::CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        for _ in 0..2 {
            backend.begin_opening_metrics().unwrap();
            let actual = backend
                .coefficient_packing_partials_batch(
                    Some(&prepared),
                    batch,
                    SubringCoefficientPackingPlan { point: &point },
                )
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual[1], actual[3]);
            let metrics = backend.last_opening_metrics().unwrap().unwrap();
            assert_eq!(metrics.cpu_fallback_calls, 0);
            assert!(metrics.command_wall_time > std::time::Duration::ZERO);
            assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
            assert_eq!(
                prepared.matrix_cache_bytes().unwrap(),
                fields.len() * size_of::<Fp128Limbs>()
            );
        }
    }
}

#[test]
fn recursive_coefficient_packing_unsupported_degree_respects_policy() {
    type E = jolt_field::FpExt4<F>;
    const D: usize = 256;
    let witness = RecursiveFoldSource::Witness(std::sync::Arc::new(
        RecursiveWitnessFlat::from_i8_digits(vec![-3; 2 * D]),
    ));
    let sources = [&witness];
    let batch =
        <RecursiveFoldSource<F> as RootOpeningSource<F, D>>::opening_batch(&sources).unwrap();
    let geometry = SubringCoefficientPackingGeometry::try_new(4, D, 64).unwrap();
    let public_point = (0..9)
        .map(|index| E::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        2,
        2,
        9,
        &public_point,
    )
    .unwrap();
    let expected = akita_prover::CpuBackend::DEFAULT
        .coefficient_packing_partials_batch(
            None,
            batch,
            SubringCoefficientPackingPlan { point: &point },
        )
        .unwrap();
    let required = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    assert!(required
        .coefficient_packing_partials_batch(
            None,
            batch,
            SubringCoefficientPackingPlan { point: &point }
        )
        .is_err());
    let preferred = MetalBackend::new(MetalExecutionPolicy::PreferMetal).unwrap();
    preferred.begin_opening_metrics().unwrap();
    let actual = preferred
        .coefficient_packing_partials_batch(
            None,
            batch,
            SubringCoefficientPackingPlan { point: &point },
        )
        .unwrap();
    assert_eq!(actual, expected);
    let metrics = preferred.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 1);
    assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
}
