use super::setup_columns::contract_setup_columns;
use super::*;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_types::CommitmentRingDims;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

type TestField = Prime128OffsetA7F7;

fn mixed_dimension_events() -> RelationWeightEvents<TestField> {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    let mut events = RelationWeightEvents {
        events: Vec::new(),
        alpha_powers: scalar_powers(TestField::from_u64(7), role_dims.d_a()),
        relation_coefficient_block_len: 32,
        physical_field_len: 256,
        setup_is_deferred: false,
    };
    events
        .push(
            0,
            128,
            0,
            TestField::from_u64(2),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
        .push(
            32,
            32,
            0,
            TestField::from_u64(3),
            RelationWeightContribution::SetupMatrix,
        )
        .unwrap();
    events
        .push(
            64,
            64,
            64,
            TestField::from_u64(5),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
        .push(
            128,
            64,
            0,
            TestField::from_u64(11),
            RelationWeightContribution::SetupMatrix,
        )
        .unwrap();
    events
        .push(
            96,
            64,
            0,
            TestField::from_u64(13),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
}

#[test]
fn mixed_dimension_factorization_reconstructs_dense_weights() {
    let events = mixed_dimension_events();
    let dense = events.materialize_dense().unwrap();
    let factorization = events.factor_common_alpha().unwrap();
    assert_eq!(factorization.common_alpha_factor().len(), 32);
    assert_eq!(
        factorization.relation_lane_weights().len(),
        dense.len() / 32
    );
    for (lane, &lane_weight) in factorization.relation_lane_weights().iter().enumerate() {
        for (coefficient, &alpha_power) in factorization.common_alpha_factor().iter().enumerate() {
            assert_eq!(
                dense[lane * factorization.common_alpha_factor().len() + coefficient],
                lane_weight * alpha_power,
            );
        }
    }
}

#[test]
fn factorization_materialization_preserves_relation_weights() {
    let events = mixed_dimension_events();
    let point = (0..8)
        .map(|index| TestField::from_u64(101 + index))
        .collect::<Vec<_>>();
    let expected_dense = events.materialize_dense().unwrap();
    let expected_factorization = events.factor_common_alpha().unwrap();
    let expected_evaluation = events.evaluate_at_point(&point, None).unwrap();
    assert_eq!(
        expected_factorization.materialize_dense().unwrap(),
        expected_dense
    );
    assert_eq!(
        events.evaluate_at_point(&point, None).unwrap(),
        expected_evaluation
    );
}

#[test]
fn factorization_rejects_an_unaligned_alpha_reset() {
    let mut events = mixed_dimension_events();
    events.events.clear();
    assert!(matches!(
        events.push(
            0,
            32,
            16,
            TestField::one(),
            RelationWeightContribution::Constraint,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn setup_columns_batch_logical_weights_over_one_physical_family() {
    let row_0 = [
        TestField::from_u64(1),
        TestField::from_u64(2),
        TestField::from_u64(3),
        TestField::from_u64(4),
    ];
    let row_1 = [
        TestField::from_u64(5),
        TestField::from_u64(6),
        TestField::from_u64(7),
        TestField::from_u64(8),
    ];
    let family = SetupRows {
        rows: vec![&row_0, &row_1],
        ring_d: 2,
    };
    let alpha = TestField::from_u64(11);
    let alpha_powers = [TestField::one(), alpha];
    let row_weights = vec![
        (0, vec![TestField::from_u64(2), TestField::from_u64(3)]),
        (1, vec![TestField::from_u64(5), TestField::zero()]),
    ];

    let evaluated = contract_setup_columns(&family, 0..2, &row_weights, 2, 1, |coefficients| {
        Ok(vec![eval_flat_ring_at_pows_fast(
            coefficients,
            &alpha_powers,
        )])
    })
    .unwrap();
    for column in 0..2 {
        let row_0_eval = row_0[2 * column] + alpha * row_0[2 * column + 1];
        let row_1_eval = row_1[2 * column] + alpha * row_1[2 * column + 1];
        assert_eq!(
            evaluated.get_scalar(0, column).unwrap(),
            TestField::from_u64(2) * row_0_eval + TestField::from_u64(5) * row_1_eval,
        );
        assert_eq!(
            evaluated.get_scalar(1, column).unwrap(),
            TestField::from_u64(3) * row_0_eval,
        );
    }
}

fn assert_direct_lane_oracle<E>()
where
    E: ExtField<jolt_field::Prime64Offset59> + Ring,
{
    use jolt_field::Prime64Offset59 as B;
    let mut state = 0x7183_9ac4_d05e_26b1u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for iteration in 0..12 {
        let alpha = match iteration {
            0 => E::zero(),
            1 => E::one(),
            _ => E::from_base_fn(|_| B::from_u64(next())),
        };
        let powers = scalar_powers(alpha, 512);
        let mut oracle = RelationWeightEvents {
            events: Vec::new(),
            alpha_powers: powers.clone(),
            relation_coefficient_block_len: 32,
            physical_field_len: 4096,
            setup_is_deferred: false,
        };
        let mut direct = events::FactoredWeightSink::new(powers, 32, 4096, false).unwrap();
        for event in 0..128 {
            let width = [32, 64, 128, 256][next() as usize % 4];
            let start = (next() as usize % ((4096 - width) / 32 + 1)) * 32;
            let exponent = (next() as usize % ((512 - width) / 32 + 1)) * 32;
            let scalar = if event % 11 == 0 {
                E::zero()
            } else {
                E::from_base_fn(|_| B::from_u64(next()))
            };
            let contribution = if event % 2 == 0 {
                RelationWeightContribution::Constraint
            } else {
                RelationWeightContribution::SetupMatrix
            };
            oracle
                .push(start, width, exponent, scalar, contribution)
                .unwrap();
            direct
                .push(start, width, exponent, scalar, contribution)
                .unwrap();
        }
        let direct = direct.finish().unwrap();
        assert_eq!(direct, oracle.factor_common_alpha().unwrap());
        assert_eq!(
            direct.materialize_dense().unwrap(),
            oracle.materialize_dense().unwrap()
        );
    }
}

#[test]
fn direct_factored_lanes_match_checked_event_oracle_degree_one_and_two() {
    assert_direct_lane_oracle::<jolt_field::Prime64Offset59>();
    assert_direct_lane_oracle::<jolt_field::Ext2<jolt_field::Prime64Offset59>>();
}

#[test]
fn direct_factored_lanes_preserve_event_rejection_and_zero_short_circuit() {
    let original = mixed_dimension_events();
    for (start, width, exponent, scalar) in [
        (1, 32, 0, TestField::one()),
        (0, 0, 0, TestField::one()),
        (0, 48, 0, TestField::one()),
        (0, 32, 1, TestField::one()),
        (256, 32, 0, TestField::one()),
        (0, 32, 128, TestField::one()),
        (usize::MAX, 32, 0, TestField::one()),
        (usize::MAX, 0, usize::MAX, TestField::zero()),
    ] {
        let mut oracle = original.clone();
        oracle.events.clear();
        let mut direct =
            events::FactoredWeightSink::new(original.alpha_powers.clone(), 32, 256, false).unwrap();
        let expected = oracle.push(
            start,
            width,
            exponent,
            scalar,
            RelationWeightContribution::Constraint,
        );
        let actual = direct.push(
            start,
            width,
            exponent,
            scalar,
            RelationWeightContribution::Constraint,
        );
        assert_eq!(actual.is_err(), expected.is_err());
        assert_eq!(
            direct.finish().unwrap(),
            oracle.factor_common_alpha().unwrap()
        );
    }
    assert!(events::FactoredWeightSink::<TestField>::new(vec![], 32, 256, false).is_err());
    assert!(events::FactoredWeightSink::<TestField>::new(
        original.alpha_powers.clone(),
        0,
        256,
        false
    )
    .is_err());
    assert!(events::FactoredWeightSink::<TestField>::new(
        original.alpha_powers.clone(),
        32,
        256,
        true
    )
    .is_err());
}

fn assert_scalar_contraction_oracle<E>()
where
    E: ExtField<jolt_field::Prime64Offset59> + MulBaseUnreduced<jolt_field::Prime64Offset59>,
{
    use jolt_field::Prime64Offset59 as B;
    let rows = (0..4)
        .map(|row| {
            (0..7 * 64)
                .map(|index| B::from_u64(7 + (index * 391 + row * 1021) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let family = SetupRows {
        rows: rows.iter().map(Vec::as_slice).collect(),
        ring_d: 64,
    };
    let weights = (0..4)
        .map(|row| {
            (
                row,
                (0..3)
                    .map(|batch| {
                        if row == batch {
                            E::zero()
                        } else {
                            E::from_base_fn(|coordinate| {
                                B::from_u64((3 + row * 13 + batch * 17 + coordinate * 19) as u64)
                            })
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let alpha = E::from_base_fn(|coordinate| B::from_u64((23 + coordinate * 29) as u64));
    let powers = scalar_powers(alpha, 64);
    let oracle = contract_setup_columns(&family, 1..7, &weights, 3, 1, |coefficients| {
        Ok(vec![eval_flat_ring_at_pows_fast(coefficients, &powers)])
    })
    .unwrap();
    let direct = contract_setup_columns(&family, 1..7, &weights, 3, 1, |coefficients| {
        Ok([eval_flat_ring_at_pows_fast(coefficients, &powers)])
    })
    .unwrap();
    for column in 0..6 {
        for batch in 0..3 {
            assert_eq!(
                direct.get_scalar(batch, column).unwrap(),
                oracle.get_scalar(batch, column).unwrap()
            );
        }
    }
    assert!(
        contract_setup_columns(&family, 0..1, &[(0, vec![])], 3, 1, |coefficients| {
            Ok([eval_flat_ring_at_pows_fast(coefficients, &powers)])
        })
        .is_err()
    );
    assert!(contract_setup_columns(
        &family,
        0..1,
        &[(4, vec![E::one()])],
        1,
        1,
        |coefficients| { Ok([eval_flat_ring_at_pows_fast(coefficients, &powers)]) }
    )
    .is_err());
}

#[test]
fn fixed_scalar_callback_matches_independent_vec_callback_oracle() {
    assert_scalar_contraction_oracle::<jolt_field::Prime64Offset59>();
    assert_scalar_contraction_oracle::<jolt_field::Ext2<jolt_field::Prime64Offset59>>();
}

fn assert_complete_evaluation_trace_compiler<E>()
where
    E: FpExtEncoding<jolt_field::Prime64Offset59>
        + Ring
        + MulBaseUnreduced<jolt_field::Prime64Offset59>,
{
    use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
    use akita_types::{
        relation_rhs_coeff_len, AkitaSetupDescriptor, CommitmentPayloadMode, DigitRangePlan,
        FlatMatrix, RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationGroupOpening,
        RingRelationMode, RingVec, SisModulusProfileId,
    };
    use jolt_field::Prime64Offset59 as B;
    const D: usize = 64;
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    params.own_group_mut().opening.num_digits_fold = 3;
    params.payload_mode = CommitmentPayloadMode::Raw;
    params.ring_relation_mode = RingRelationMode::QuotientLift;
    let batch = OpeningClaimsLayout::new(8, 1).unwrap();
    let group = params.group_params(&batch, 0).unwrap();
    let blocks = group.num_live_blocks();
    let challenges = Challenges::from_sparse(
        (0..blocks)
            .map(|block| SparseChallenge {
                positions: vec![(block % D) as u32].into(),
                coeffs: vec![if block % 2 == 0 { 1 } else { -1 }].into(),
            })
            .collect(),
        blocks,
        1,
    )
    .unwrap();
    let multiplier = RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
        position_weights: (0..group.num_positions_per_block())
            .map(|position| B::from_u64(2 + position as u64))
            .collect(),
        live_block_weights: vec![B::zero(); blocks],
    });
    let relation_geometry = RelationWitnessGeometry::for_level(&params, &batch, E::DEGREE).unwrap();
    let mut gamma_ring = vec![B::zero(); D];
    gamma_ring[0] = B::from_u64(7);
    let relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::evaluation_trace(
            challenges, multiplier,
        )],
        E::DEGREE,
        batch.clone(),
        vec![B::from_u64(7)],
        RingVec::from_coeffs_with_ring_dim(gamma_ring, D).unwrap(),
        RingVec::from_coeffs(vec![
            B::zero();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())
                .unwrap()
        ]),
        RingVec::from_coeffs(Vec::new()),
        params.role_dims(),
    )
    .unwrap();
    let witness_layout = relation.segment_layout(&params, None).unwrap();
    let address = RelationAddressGeometry::for_relation(
        &relation_geometry,
        D,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let plan = RelationRangeImagePlan::new(
        relation_geometry,
        address,
        DigitRangePlan::new(1 << params.open().digits.log_basis).unwrap(),
        witness_layout,
        &batch,
    )
    .unwrap();
    let setup_field_len = 1usize << 18;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            num_field_elements: setup_field_len,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_field_len)
                .map(|index| B::from_u64(1 + (index as u64).wrapping_mul(391)))
                .collect(),
        ),
    );
    for seed in 0..3u64 {
        let alpha =
            E::from_base_fn(|coordinate| B::from_u64(23 + seed * 31 + coordinate as u64 * 37));
        let tau1 = (0..plan.relation_row_index_num_vars().unwrap())
            .map(|index| {
                E::from_base_fn(|coordinate| {
                    B::from_u64(41 + seed * 43 + index as u64 * 47 + coordinate as u64 * 53)
                })
            })
            .collect::<Vec<_>>();
        let gamma = vec![E::from_base_fn(|coordinate| {
            B::from_u64(61 + seed * 67 + coordinate as u64 * 71)
        })];
        let inputs = || RelationWeightEventInputs {
            setup: RelationSetupSource::Matrix(&setup),
            instance: &relation,
            alpha,
            level_params: &params,
            relation_row_point: &tau1,
            claim_coefficients: &gamma,
            opening_source_len: plan.digit_witness_domain().domain_len() / D,
            opening_ring_dim: D,
            relation_plan: &plan,
            opening_points: OpeningFamily::EvaluationTrace(()),
        };
        let (oracle, oracle_semantics) = build_relation_weight_events(inputs()).unwrap();
        let (direct, direct_semantics) =
            compile_relation_weight_factorization(inputs(), None).unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let fallback =
            |source: &AkitaExpandedSetup<B>,
             projections: &[crate::compute::SetupColumnContractionPlan<'_, E>]| {
                assert!(std::ptr::eq(source, &setup));
                assert_eq!(projections.len(), 3);
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            };
        let (fallback_direct, fallback_semantics) =
            compile_relation_weight_factorization(inputs(), Some(&fallback)).unwrap();
        assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert_eq!(fallback_direct, direct);
        assert_eq!(fallback_semantics, direct_semantics);

        assert_eq!(direct, oracle.factor_common_alpha().unwrap());
        assert_eq!(
            direct.materialize_dense().unwrap(),
            oracle.materialize_dense().unwrap()
        );
        assert_eq!(direct_semantics, oracle_semantics);
    }
}

#[test]
fn direct_complete_trace_compiler_matches_reference_degree_one_and_two() {
    assert_complete_evaluation_trace_compiler::<jolt_field::Prime64Offset59>();
    assert_complete_evaluation_trace_compiler::<jolt_field::Ext2<jolt_field::Prime64Offset59>>();
}
