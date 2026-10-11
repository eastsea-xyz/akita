use super::*;
use akita_prover::ComputeBackendSetup;
use akita_types::{
    sample_akita_setup_seed, AkitaCommitmentHint, AkitaExpandedSetup, AkitaSetupDescriptor,
    CommittedGroupParams, FlatMatrix, GroupCommitPhaseParams, InnerCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout, SetupPrefixPublicCommitment, SetupPrefixSlot,
    SetupPrefixSlotId, SisModulusProfileId,
};

const D: usize = 128;

fn fixture() -> (MetalBackend, MetalPreparedSetup, ProverSource<F>) {
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
    // The nonzero physical suffix past natural_len=400 must contribute.
    let fields = (0..512)
        .map(|i| F::from_i64((i % 17) as i64 - 8))
        .collect::<Vec<_>>();
    let expanded = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 9,
                max_num_batched_polys: 1,
                num_field_elements: fields.len(),
                setup_seed: sample_akita_setup_seed(),
            },
            FlatMatrix::from_flat_data(fields),
        ),
    );
    let slot = Arc::new(SetupPrefixSlot {
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
    let prepared = backend.prepare_setup(&setup).unwrap();
    (
        backend,
        prepared,
        ProverSource::SetupPrefix { expanded, slot },
    )
}

#[test]
fn recursive_setup_prefix_frozen_slot_evaluation_and_decomposition_match_cpu() {
    let (backend, prepared, source) = fixture();
    let view = <ProverSource<F> as RootOpeningSource<F, D>>::opening_view(&source).unwrap();
    let positions = [2, 3, 5, 7].map(F::from_u64);
    let folds = [F::from_u64(11)];
    let plan = OpeningFoldPlan::Base {
        live_block_weights: &folds,
        position_weights: &positions,
        num_positions_per_block: 4,
    };
    let expected =
        OpeningFoldKernel::evaluate_and_fold(&CpuBackend::DEFAULT, Some(&prepared.cpu), view, plan)
            .unwrap();
    for _ in 0..2 {
        backend.begin_opening_metrics().unwrap();
        assert_eq!(
            OpeningFoldKernel::evaluate_and_fold(&backend, Some(&prepared), view, plan).unwrap(),
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
    let alpha_bits = D.trailing_zeros() as usize;
    let mut point = vec![Ext2::<F>::zero(); alpha_bits + 2];
    point[alpha_bits] = Ext2::from_base_slice(&[F::from_u64(2), F::from_u64(3)]);
    point[alpha_bits + 1] = Ext2::from_base_slice(&[F::from_u64(5), F::from_u64(7)]);
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let point =
            prepare_opening_point::<F, Ext2<F>, D>(&point, basis, 4, 1, alpha_bits).unwrap();
        let plan = OpeningFoldPlan::Subfield {
            multipliers: point.ring_multiplier_point.as_subfield().unwrap(),
            num_positions_per_block: 4,
        };
        let expected = OpeningFoldKernel::evaluate_and_fold(
            &CpuBackend::DEFAULT,
            Some(&prepared.cpu),
            view,
            plan,
        )
        .unwrap();
        assert_eq!(
            OpeningFoldKernel::evaluate_and_fold(&backend, Some(&prepared), view, plan).unwrap(),
            expected
        );
    }
    let challenge = SparseChallenge {
        positions: [0, 127, 37].into_iter().collect(),
        coeffs: [1, -3, 2].into_iter().collect(),
    };
    let plan = DecomposeFoldPlan {
        challenges: std::slice::from_ref(&challenge),
        num_positions_per_block: 4,
        num_digits: 4,
        log_basis: 4,
    };
    let expected =
        OpeningFoldKernel::decompose_fold(&CpuBackend::DEFAULT, Some(&prepared.cpu), view, plan)
            .unwrap();
    backend.begin_opening_metrics().unwrap();
    assert_eq!(
        OpeningFoldKernel::decompose_fold(&backend, Some(&prepared), view, plan).unwrap(),
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

#[test]
fn recursive_mixed_setup_prefix_batch_preserves_selected_claim_order() {
    let (backend, prepared, prefix) = fixture();
    let first = ProverSource::Witness(Arc::new(RecursiveWitnessFlat::from_i8_digits(
        (0..512).map(|i| (i % 7) as i8 - 3).collect(),
    )));
    let second = ProverSource::Witness(Arc::new(RecursiveWitnessFlat::from_i8_digits(
        (0..512).map(|i| (i % 11) as i8 - 5).collect(),
    )));
    let sources = [&second, &prefix, &first, &prefix, &second];
    let challenges = (0..sources.len())
        .map(|claim| SparseChallenge {
            positions: [0, 127, 37].into_iter().collect(),
            coeffs: [claim as i8 + 1, -(claim as i8) - 2, 3]
                .into_iter()
                .collect(),
        })
        .collect::<Vec<_>>();
    let mut expected = vec![0i32; 4 * D];
    for (source, challenge) in sources.iter().zip(&challenges) {
        let view = <ProverSource<F> as RootOpeningSource<F, D>>::opening_view(source).unwrap();
        let output = OpeningFoldKernel::decompose_fold(
            &CpuBackend::DEFAULT,
            Some(&prepared.cpu),
            view,
            DecomposeFoldPlan {
                challenges: std::slice::from_ref(challenge),
                num_positions_per_block: 4,
                num_digits: 1,
                log_basis: 4,
            },
        )
        .unwrap();
        for (dst, &value) in expected.iter_mut().zip(output.centered_coeffs_flat()) {
            *dst += value;
        }
    }
    let batch = <ProverSource<F> as RootOpeningSource<F, D>>::opening_batch(&sources).unwrap();
    for _ in 0..2 {
        backend.begin_opening_metrics().unwrap();
        let output = OpeningBatchKernel::decompose_fold_batch(
            &backend,
            Some(&prepared),
            batch,
            DecomposeFoldBatchPlan::Sparse {
                challenges: &challenges,
                num_positions_per_block: 4,
                num_digits: 1,
                log_basis: 4,
            },
        )
        .unwrap();
        let BatchDecomposeFoldOutcome::Fused(output) = output else {
            panic!("expected fused mixed-source batch");
        };
        assert_eq!(output.centered_coeffs_flat(), expected);
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

#[test]
fn recursive_setup_prefix_rejects_frozen_geometry_before_source_work() {
    let (backend, prepared, prefix) = fixture();
    let view = <ProverSource<F> as RootOpeningSource<F, D>>::opening_view(&prefix).unwrap();
    let positions = [F::one(); 2];
    let folds = [F::one(); 2];
    backend.begin_opening_metrics().unwrap();
    let wrong_positions = OpeningFoldKernel::evaluate_and_fold(
        &backend,
        Some(&prepared),
        view,
        OpeningFoldPlan::Base {
            live_block_weights: &folds,
            position_weights: &positions,
            num_positions_per_block: 2,
        },
    );
    assert!(wrong_positions.is_err());
    let wrong_ring = <ProverSource<F> as RootOpeningSource<F, 64>>::opening_view(&prefix).unwrap();
    let positions = [F::one(); 4];
    let folds = [F::one()];
    assert!(OpeningFoldKernel::evaluate_and_fold(
        &backend,
        Some(&prepared),
        wrong_ring,
        OpeningFoldPlan::Base {
            live_block_weights: &folds,
            position_weights: &positions,
            num_positions_per_block: 4
        }
    )
    .is_err());
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
    assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
}
