//! Prover setup artifact and config-free setup expansion helpers.

use akita_error::AkitaError;

use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor,
    AkitaSetupSeed, AkitaVerifierSetup, FlatMatrix, SetupMatrixCapacity, SetupPrefixProverRegistry,
    SetupPrefixVerifierRegistry,
};
use jolt_field::{CanonicalEncoding, Field};
use std::sync::Arc;

/// Trusted producer of the exact public field prefix for one setup seed.
///
/// A borrowed producer is `Sync` because setup construction can enter the
/// caller's Rayon pool. Shape and field encodings are checked at construction;
/// seed derivation is the producer's contract and is not re-executed on CPU.
pub type SetupMatrixGenerator<'a, F> =
    dyn Fn(usize, &AkitaSetupSeed) -> Result<FlatMatrix<F>, AkitaError> + Sync + 'a;

/// Prover setup artifact.
///
/// Backend-prepared compute state is intentionally not stored here. Host code
/// prepares a compute backend from the expanded setup when it wants to prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: Field> {
    /// Expanded matrix stage used by the prover.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Preprocessed setup-prefix commitment slots for setup-claim offloading.
    ///
    /// D-free (S4): the registry stores flat ring-coefficient commitment rows and
    /// D-free hints; concrete-D selection happens at backend-prepare /
    /// per-operation time, not on this artifact.
    pub prefix_slots: SetupPrefixProverRegistry<F>,
}

impl<F: Field> AkitaProverSetup<F> {
    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived provisioning bounds in base-field
    /// elements. This constructor owns only the concrete prover artifact:
    /// materialization of that prefix of the public field stream.
    /// `None` uses the canonical CPU producer; a supplied trusted producer
    /// replaces only matrix expansion, preserving setup identity and layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity is invalid or the setup descriptor
    /// cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        setup_capacity: SetupMatrixCapacity,
        matrix_generator: Option<&SetupMatrixGenerator<'_, F>>,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding + AkitaSerialize,
    {
        let setup_seed = sample_akita_setup_seed();
        let seed = AkitaSetupDescriptor {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements: setup_capacity.num_field_elements,
            setup_seed: setup_seed.clone(),
        };
        seed.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {err}"))
        })?;

        let shared_flat = match matrix_generator {
            Some(generate) => {
                let matrix = generate(setup_capacity.num_field_elements, &setup_seed)?;
                if matrix.num_field_elements() != setup_capacity.num_field_elements {
                    return Err(AkitaError::InvalidSetup(format!(
                        "setup matrix generator returned {} field elements, expected {}",
                        matrix.num_field_elements(),
                        setup_capacity.num_field_elements,
                    )));
                }
                let mut encoded = vec![0u8; F::NUM_BYTES];
                for (index, value) in matrix.as_field_slice().iter().enumerate() {
                    value.to_bytes_le(&mut encoded);
                    if F::from_bytes_le_checked(&encoded) != Some(*value) {
                        return Err(AkitaError::InvalidSetup(format!(
                            "setup matrix generator returned a noncanonical field at index {index}",
                        )));
                    }
                }
                matrix
            }
            None => {
                derive_public_matrix_prefix::<F>(setup_capacity.num_field_elements, &setup_seed)
            }
        };
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_flat),
        );

        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(setup_seed),
        })
    }

    /// Derive a verifier setup with an explicit public-matrix capacity.
    ///
    /// The verifier keeps only the public stream prefix needed by its direct
    /// setup scans and terminal matrix. The setup-prefix registry remains
    /// complete because offloaded claims can refer to entries beyond that
    /// materialized matrix prefix. Verifier setup also initializes a
    /// non-serialized lazy terminal NTT-prefix cache.
    ///
    /// # Errors
    ///
    /// Returns an error if `matrix_capacity` is empty or exceeds the prover
    /// matrix, or if prover prefix-slot metadata cannot be converted into
    /// verifier-visible prefix slots.
    pub fn to_verifier_setup(
        &self,
        matrix_capacity: SetupMatrixCapacity,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError>
    where
        F: Valid,
    {
        if matrix_capacity.num_field_elements == 0 {
            return Err(AkitaError::InvalidSetup(
                "verifier setup matrix capacity must be non-zero".to_string(),
            ));
        }
        let prover_matrix = self.expanded.shared_matrix().as_field_slice();
        let verifier_matrix = prover_matrix
            .get(..matrix_capacity.num_field_elements)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "verifier setup requires {} field elements but prover setup has {}",
                    matrix_capacity.num_field_elements,
                    prover_matrix.len()
                ))
            })?;
        let expanded = if verifier_matrix.len() == prover_matrix.len() {
            self.expanded.clone()
        } else {
            let mut descriptor = self.expanded.descriptor().clone();
            descriptor.num_field_elements = verifier_matrix.len();
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    FlatMatrix::from_flat_data(verifier_matrix.to_vec()),
                ),
            )
        };
        let mut prefix_slots =
            SetupPrefixVerifierRegistry::new(self.expanded.descriptor().setup_seed.clone());
        prefix_slots.replace_from_prover_registry(&self.prefix_slots)?;
        AkitaVerifierSetup::from_parts(expanded, prefix_slots)
    }

    /// Wrap an already-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// Use this when the caller has already run strict setup validation, for
    /// example through checked setup deserialization. This still re-checks
    /// seed-to-matrix derivation at the trust boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup does not match its seed.
    pub fn from_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding + Valid,
    {
        expanded.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup validation failed: {err}"))
        })?;
        Self::from_seed_validated_expanded(expanded)
    }

    /// Wrap a seed-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// This skips seed-to-matrix rederivation. Use it only when the caller
    /// just verified the matrix with `validate_public_matrix_matches_seed` in
    /// the same trust boundary, such as the disk-cache loader in
    /// `akita-setup`.
    ///
    /// # Errors
    ///
    /// Returns an error if the seed and matrix disagree or their internal shape
    /// metadata is malformed.
    pub fn from_seed_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding + Valid,
    {
        expanded.descriptor().check().map_err(|err| {
            AkitaError::InvalidSetup(format!(
                "expanded setup descriptor validation failed: {err}"
            ))
        })?;
        expanded.shared_matrix().check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup matrix validation failed: {err}"))
        })?;
        if expanded.shared_matrix().num_field_elements() != expanded.descriptor().num_field_elements
        {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix field count does not match setup descriptor".to_string(),
            ));
        }
        let setup_seed = expanded.descriptor().setup_seed.clone();
        let expanded = Arc::new(expanded);
        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(setup_seed),
        })
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> Valid for AkitaProverSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.setup_seed() != &self.expanded.descriptor().setup_seed {
            return Err(SerializationError::InvalidData(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        self.prefix_slots.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;

    #[test]
    fn generate_with_capacity_rejects_zero_setup_len() {
        let zero_len = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 0,
            },
            None,
        )
        .expect_err("zero setup length must not produce an undecodable setup");
        assert!(zero_len.to_string().contains("num_field_elements"));
    }

    #[test]
    fn verifier_setup_conversion_enforces_the_requested_prefix() {
        let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 8,
            },
            None,
        )
        .expect("generate setup");

        let verifier = setup
            .to_verifier_setup(SetupMatrixCapacity {
                num_field_elements: 3,
            })
            .expect("narrow verifier setup");
        assert_eq!(verifier.expanded.descriptor().num_field_elements, 3);
        assert_eq!(verifier.expanded.shared_matrix().num_field_elements(), 3);
        assert_eq!(
            verifier.expanded.shared_matrix().as_field_slice(),
            &setup.expanded.shared_matrix().as_field_slice()[..3]
        );
        assert!(
            setup
                .to_verifier_setup(SetupMatrixCapacity {
                    num_field_elements: 9,
                })
                .is_err(),
            "conversion must not extend beyond the prover matrix"
        );
    }

    #[test]
    fn prover_setup_check_validates_prefix_slots() {
        use akita_types::{
            scheduled_setup_prefix, AkitaCommitmentHint, CompressionChainPlan,
            CompressionChainWitness, GroupCommitPhaseParams, GroupOpenPhaseParams,
            InnerCommitMatrixParams, OuterCommitMatrixParams, PackedNegativeBinary,
            PolynomialGroupLayout, RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot,
            SisMatrixRole, SisModulusProfileId, SisTableDigest, SisTableKey,
            DEFAULT_SIS_SECURITY_POLICY,
        };

        const PREFIX_D: usize = 64;

        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity::minimum(),
            None,
        )
        .expect("generate setup");
        let decomposed =
            RingVec::from_coeffs_with_ring_dim(Vec::new(), PREFIX_D).expect("empty A-native hint");
        let inner_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            PREFIX_D,
            3,
            &akita_challenges::SparseChallengeConfig::production_for_ring_dim(PREFIX_D)
                .expect("D=64 has a production challenge configuration"),
            1,
            1,
        )
        .expect("audited prefix A bound");
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Inner,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: inner_bound,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Outer,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank(),
        )
        .expect("audited prefix B matrix");
        let commitment_rows = outer_commit_matrix.output_rank();
        let compression_plan = CompressionChainPlan::for_complete_source(
            SisModulusProfileId::Q128OffsetA7F7,
            commitment_rows * PREFIX_D,
        )
        .expect("prefix compression plan");
        let compression_stages = compression_plan
            .maps()
            .iter()
            .map(|map| PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()]))
            .collect::<Result<Vec<_>, _>>()
            .expect("zero compression stages");
        let compression_witness =
            CompressionChainWitness::new(compression_plan.clone(), compression_stages)
                .expect("zero compression witness");
        let compression_quotients = compression_plan
            .maps()
            .iter()
            .map(|map| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![Prime128Offset275::default(); map.output_coefficients()],
                    map.ring_dimension(),
                )
                .expect("zero compression quotient")
            })
            .collect::<Vec<_>>();
        let hint = AkitaCommitmentHint::singleton_with_outer_compression(
            decomposed,
            &compression_witness,
            &compression_quotients,
        )
        .expect("compression-valid A-native hint");
        let commitment_params = GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: GroupCommitPhaseParams {
                version: GroupCommitPhaseParams::VERSION,
                group: PolynomialGroupLayout::singleton(PREFIX_D.trailing_zeros() as usize),
                blocks: akita_types::BlockGeometry::new(1, 1, 1),
                outer_slice_count: akita_types::CommitmentSliceCount::ONE,
                inner: akita_types::RoleParams::new(
                    akita_types::GadgetDigits::new(1, 1),
                    inner_commit_matrix,
                ),
                outer: akita_types::RoleParams::new(
                    akita_types::GadgetDigits::new(1, 1),
                    outer_commit_matrix,
                ),
            },
            opening: akita_types::GroupOpeningPlan::evaluation_trace(
                akita_challenges::SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        };
        let err = setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: scheduled_setup_prefix(PREFIX_D, commitment_params)
                    .slot_id()
                    .expect("setup prefix group"),
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![
                        RingVec::from_coeffs(vec![Prime128Offset275::default(); PREFIX_D]);
                        commitment_rows
                    ],
                },
                hint,
            })
            .expect_err("insert must reject malformed prefix slot");
        assert!(err.to_string().contains("commitment row"));
    }

    #[test]
    fn trusted_matrix_generator_matches_the_canonical_cpu_setup() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = AtomicUsize::new(0);
        let generate: &SetupMatrixGenerator<'_, Prime128Offset275> = &|count, seed| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(count, 64);
            assert_eq!(seed, &sample_akita_setup_seed());
            Ok(derive_public_matrix_prefix(count, seed))
        };
        let capacity = SetupMatrixCapacity {
            num_field_elements: 64,
        };
        let supplied = AkitaProverSetup::generate_with_capacity(8, 1, capacity, Some(generate))
            .expect("trusted matrix producer");
        let cpu =
            AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(8, 1, capacity, None)
                .expect("canonical CPU producer");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(supplied, cpu);
    }

    #[test]
    fn trusted_matrix_generator_rejects_wrong_field_capacity() {
        for returned_count in [63, 65] {
            let generate: &SetupMatrixGenerator<'_, Prime128Offset275> =
                &|_, seed| Ok(derive_public_matrix_prefix(returned_count, seed));
            let error = AkitaProverSetup::generate_with_capacity(
                8,
                1,
                SetupMatrixCapacity {
                    num_field_elements: 64,
                },
                Some(generate),
            )
            .expect_err("mismatched producer extent must fail before setup assembly");
            assert!(error.to_string().contains("field elements, expected 64"));
        }
    }

    #[test]
    fn trusted_matrix_generator_errors_are_returned_without_cpu_rederivation() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = AtomicUsize::new(0);
        let generate: &SetupMatrixGenerator<'_, Prime128Offset275> = &|_, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(AkitaError::InvalidSetup("forced producer error".to_owned()))
        };
        let error = AkitaProverSetup::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 64,
            },
            Some(generate),
        )
        .expect_err("producer error must be returned before CPU fallback is chosen by the host");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            matches!(error, AkitaError::InvalidSetup(message) if message == "forced producer error")
        );
    }
}
