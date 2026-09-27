//! Verifier-only setup derivation.
//!
//! [`crate::new_prover_setup`] materializes the whole public matrix the
//! prover commits with, and [`akita_prover::AkitaProverSetup::to_verifier_setup`]
//! then cuts a verifier setup out of it. A verifier needs only two things from
//! that artifact:
//!
//! 1. the public-matrix prefix its direct setup scans and terminal matrix read
//!    ([`SetupRequirements::verifier_matrix_capacity`]), which is a prefix of
//!    the seed-derived public stream and is derived here directly, and
//! 2. the setup-prefix commitments of setup-claim offloading. Each is a SIS
//!    commitment to a power-of-two prefix of the same public stream, which can
//!    be as long as the prover matrix itself. There is no shortcut to them
//!    short of committing that prefix: they exist precisely so the verifier
//!    never scans it. They are therefore taken from a caller-supplied registry
//!    of previously derived commitments, and only the ones missing from it are
//!    derived here, from a public-matrix prefix just large enough for them.
//!
//! The result equals the prover-derived verifier setup narrowed to the same
//! capacity, field element for field element.

use crate::recursive_prefixes;
use akita_config::{CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_prover::AkitaProverSetup;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, setup_prefix_slot_field_elements,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup, SetupMatrixCapacity,
    SetupPrefixSlotId, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced};
use std::sync::Arc;

/// Verifier setup for a root commitment config, derived without the prover
/// matrix.
///
/// `known_prefix_slots` supplies setup-prefix commitments derived earlier from
/// the same public stream (for example, shipped with the verifier). Each is
/// used only if it is one of this setup's required slots; required slots it
/// lacks are derived here, which costs a commitment over that prefix of the
/// public stream. Pass an empty registry to derive every slot.
///
/// # Errors
///
/// Returns an error if the capacity is invalid, `known_prefix_slots` belongs
/// to a different public stream, or deriving a missing slot fails.
#[tracing::instrument(
    skip_all,
    name = "new_verifier_setup",
    fields(
        max_num_vars,
        max_num_batched_polys,
        verifier_matrix_fields = tracing::field::Empty,
        derived_prefix_slots = tracing::field::Empty,
    )
)]
pub fn new_verifier_setup<F, Cfg>(
    schedules: &TrustedScheduleCatalog<Cfg>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    known_prefix_slots: &SetupPrefixVerifierRegistry<F>,
) -> Result<AkitaVerifierSetup<F>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg: CommitmentConfig<Field = F>,
{
    let requirements =
        SetupRequirements::from_catalog::<Cfg>(schedules, max_num_vars, max_num_batched_polys)?;
    let capacity = verifier_matrix_capacity(&requirements);
    let _ = tracing::Span::current().record("verifier_matrix_fields", capacity.num_field_elements);

    let setup_seed = sample_akita_setup_seed();
    if known_prefix_slots.setup_seed() != &setup_seed {
        return Err(AkitaError::InvalidSetup(
            "known setup-prefix commitments belong to a different public matrix".to_string(),
        ));
    }
    let descriptor = AkitaSetupDescriptor {
        max_num_vars,
        max_num_batched_polys,
        num_field_elements: capacity.num_field_elements,
        setup_seed: setup_seed.clone(),
    };
    descriptor
        .check()
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed validation failed: {err}")))?;

    let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed);
    let mut missing = Vec::new();
    for id in &requirements.prefix_slot_ids {
        match known_prefix_slots.get(id) {
            Some(slot) => prefix_slots.insert(slot.clone().with_committed_ring_dimension()?)?,
            None => missing.push(id.clone()),
        }
    }
    let _ = tracing::Span::current().record("derived_prefix_slots", missing.len());
    if !missing.is_empty() {
        tracing::warn!(
            missing = missing.len(),
            "deriving setup-prefix commitments that were not supplied; this commits their \
             public-matrix prefixes"
        );
        for slot in derive_setup_prefix_verifier_slots::<F>(&missing)? {
            prefix_slots.insert(slot)?;
        }
    }

    let shared_matrix =
        derive_public_matrix_prefix::<F>(capacity.num_field_elements, &descriptor.setup_seed);
    let expanded = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, shared_matrix),
    );
    AkitaVerifierSetup::from_parts(expanded, prefix_slots)
}

/// The public-matrix capacity [`new_verifier_setup`] materializes: the
/// verifier requirement, never more than the prover matrix it narrows.
#[must_use]
pub fn verifier_matrix_capacity(requirements: &SetupRequirements) -> SetupMatrixCapacity {
    SetupMatrixCapacity {
        num_field_elements: requirements
            .verifier_matrix_capacity
            .num_field_elements
            .min(requirements.matrix_capacity.num_field_elements)
            .max(1),
    }
}

/// Setup-prefix slot ids a setup for this catalog and capacity requires, in
/// canonical order.
///
/// # Errors
///
/// Returns an error if the capacity is invalid for the catalog.
pub fn required_setup_prefix_slot_ids<F, Cfg>(
    schedules: &TrustedScheduleCatalog<Cfg>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError>
where
    F: Field,
    Cfg: CommitmentConfig<Field = F>,
{
    Ok(
        SetupRequirements::from_catalog::<Cfg>(schedules, max_num_vars, max_num_batched_polys)?
            .prefix_slot_ids,
    )
}

/// Derive the verifier half of setup-prefix commitments from the public
/// stream, exactly as prover setup construction commits them.
///
/// The public-matrix prefix materialized is the largest footprint among
/// `slot_ids` (each slot's committed prefix and its commitment matrices), not
/// a prover setup's full matrix.
///
/// # Errors
///
/// Returns an error if a slot id is malformed or committing it fails.
pub fn derive_setup_prefix_verifier_slots<F>(
    slot_ids: &[SetupPrefixSlotId],
) -> Result<Vec<SetupPrefixVerifierSlot<F>>, AkitaError>
where
    F: Field + CanonicalEncoding + Unreduced + Valid + AkitaSerialize + 'static,
{
    if slot_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut num_field_elements = 1usize;
    for id in slot_ids {
        num_field_elements = num_field_elements.max(setup_prefix_slot_field_elements(id)?);
    }
    // The descriptor's provisioning bounds are metadata of the prover artifact
    // and do not enter a prefix commitment.
    let mut setup = AkitaProverSetup::<F>::generate_with_capacity(
        0,
        1,
        SetupMatrixCapacity { num_field_elements },
    )?;
    recursive_prefixes::populate_required_setup_prefix_slots(&mut setup, slot_ids)?;
    Ok(slot_ids
        .iter()
        .filter_map(|id| setup.prefix_slots.get(id))
        .map(akita_types::SetupPrefixSlot::verifier_slot)
        .collect())
}
