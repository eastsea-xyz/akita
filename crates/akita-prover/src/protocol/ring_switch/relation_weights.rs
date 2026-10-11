//! Semantic relation-weight events and their canonical consumers.

#[path = "relation_weights/compiler.rs"]
mod compiler;
#[path = "relation_weights/reduced_dense.rs"]
mod reduced_dense;
#[path = "relation_weights/setup_columns.rs"]
mod setup_columns;
#[path = "relation_weights/setup_projection.rs"]
mod setup_projection;
pub(crate) use setup_projection::SetupColumnProjection;
use setup_projection::{contract_scalar_setup_batch, ScalarSetupRequest};

use std::ops::Range;

use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::ring::scalar_powers;
use akita_error::AkitaError;
use akita_types::{
    gadget_row_scalars, prepare_coefficient_packing_batch_semantics, r_decomp_levels,
    AkitaExpandedSetup, CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout, OpeningFamily, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, RelationAddressGeometry, RelationRangeImagePlan,
    RelationRowFamily, RelationWitnessGeometry, RingRelationInstance, SetupProjectionGeometry,
};
pub use akita_types::{RelationWeightContribution, RelationWeightEvent};
use compiler::{
    compile_group_et_addresses, compile_group_z_addresses, EtWeightSink, RelationWeightCompilation,
    ZWeightSink,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use setup_columns::{contract_setup_residue_columns, SetupColumnValues, SetupRows};

/// Source of setup-matrix relation weights for this evaluation.
#[derive(Clone, Copy)]
pub enum RelationSetupSource<'a, F: Field> {
    /// Emit setup events directly from the expanded setup matrix.
    Matrix(&'a AkitaExpandedSetup<F>),
    /// Omit setup events because their complete evaluation is supplied separately.
    DeferredClaim,
}

/// Inputs to the one semantic relation-event builder.
pub struct RelationWeightEventInputs<'a, F: Field, E: Field> {
    pub setup: RelationSetupSource<'a, F>,
    pub instance: &'a RingRelationInstance<F>,
    pub alpha: E,
    pub level_params: &'a CommittedGroupParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
    pub relation_plan: &'a RelationRangeImagePlan,
    /// Method-typed prepared points for the current fold.
    pub opening_points:
        OpeningFamily<(), &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<E>)]>,
}

mod events;
use events::{FactoredWeightSink, RelationWeightTarget};
pub use events::{RelationWeightEvents, RelationWeightFactorization};
pub(super) use reduced_dense::build_reduced_dense_relation_weights;

fn relation_d_group_width(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
    let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
    let opening_width = relation_geometry
        .group_opening_geometry(group_index)?
        .physical_coefficient_width();
    let d_subcolumns = opening_width
        .checked_div(group_dims.d_d())
        .filter(|count| *count > 0 && opening_width.is_multiple_of(group_dims.d_d()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("opening width does not factor the D role".into())
        })?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .and_then(|n| n.checked_mul(d_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_d_column_ranges(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
) -> Result<Vec<Range<usize>>, AkitaError> {
    let mut cursor = 0usize;
    let mut seen = vec![false; opening_batch.num_groups()];
    let mut ranges = vec![0..0; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, relation_geometry, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        ranges[group_id] = cursor..end;
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(ranges)
}

fn matching_row_range(
    row_families: &[RelationRowFamily],
    mut matches: impl FnMut(&RelationRowFamily) -> bool,
) -> Result<Range<usize>, AkitaError> {
    let mut matched = row_families
        .iter()
        .enumerate()
        .filter_map(|(row, family)| matches(family).then_some(row));
    let start = matched.next().ok_or(AkitaError::InvalidProof)?;
    let mut end = start + 1;
    for row in matched {
        if row != end {
            return Err(AkitaError::InvalidSetup(
                "relation row family is not contiguous".into(),
            ));
        }
        end += 1;
    }
    Ok(start..end)
}

#[derive(Clone, Copy)]
enum LiftedEtSetup<'a, E: Field> {
    Matrix {
        d: &'a SetupColumnValues<E>,
        b: &'a SetupColumnValues<E>,
    },
    Deferred,
}

struct LiftedEtSink<'a, E: Field, S: RelationWeightTarget<E>> {
    events: &'a mut S,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    challenge_evaluations: &'a [E],
    setup: LiftedEtSetup<'a, E>,
}

impl<E: Field, S: RelationWeightTarget<E>> EtWeightSink<E> for LiftedEtSink<'_, E, S> {
    fn add_e(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            self.events.push(
                physical_start,
                self.plan.roles.d_d,
                role_subcolumn * self.plan.roles.d_d,
                self.challenge_evaluations
                    .get(challenge_index)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?
                    * constraint_scale,
                RelationWeightContribution::Constraint,
            )?;
        }
        if let LiftedEtSetup::Matrix { d, .. } = self.setup {
            self.events.push(
                physical_start,
                self.plan.roles.d_d,
                0,
                d.get_scalar(0, setup_column)?,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }

    fn add_t(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        slice_index: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        self.events.push(
            physical_start,
            self.plan.roles.d_b,
            role_subcolumn * self.plan.roles.d_b,
            self.challenge_evaluations
                .get(challenge_index)
                .copied()
                .ok_or(AkitaError::InvalidProof)?
                * constraint_scale,
            RelationWeightContribution::Constraint,
        )?;
        if let LiftedEtSetup::Matrix { b, .. } = self.setup {
            self.events.push(
                physical_start,
                self.plan.roles.d_b,
                0,
                b.get_scalar(slice_index, setup_column)?,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum LiftedZSetup<'a, E: Field> {
    Matrix(&'a SetupColumnValues<E>),
    Deferred,
}

struct LiftedZSink<'a, E: Field, S: RelationWeightTarget<E>> {
    events: &'a mut S,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    opening_evaluations: &'a [E],
    setup: LiftedZSetup<'a, E>,
}

impl<E: Field, S: RelationWeightTarget<E>> ZWeightSink<E> for LiftedZSink<'_, E, S> {
    fn add_z(
        &mut self,
        physical_start: usize,
        position: usize,
        setup_column: usize,
        constraint_scale: E,
        setup_scale: E,
    ) -> Result<(), AkitaError> {
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            self.events.push_native_ring(
                physical_start,
                self.plan.roles.d_a,
                self.opening_evaluations
                    .get(position)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?
                    * constraint_scale,
                RelationWeightContribution::Constraint,
            )?;
        }
        if let LiftedZSetup::Matrix(setup) = self.setup {
            self.events.push_native_ring(
                physical_start,
                self.plan.roles.d_a,
                setup.get_scalar(0, setup_column)? * setup_scale,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }
}

/// Emit the complete checked relation semantics for one fold.
pub(super) type RelationWeightBuild<E, S = RelationWeightEvents<E>> =
    (S, OpeningFamily<(), CoefficientPackingBatchSemantics<E>>);

#[tracing::instrument(skip_all, name = "build_relation_weight_events")]
pub fn build_relation_weight_events<F, E>(
    inputs: RelationWeightEventInputs<'_, F, E>,
) -> Result<RelationWeightBuild<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    build_relation_weights(
        inputs,
        |alpha_powers, relation_coefficient_block_len, physical_field_len, setup_is_deferred| {
            Ok(RelationWeightEvents {
                events: Vec::new(),
                alpha_powers,
                relation_coefficient_block_len,
                physical_field_len,
                setup_is_deferred,
            })
        },
        None,
    )
}

/// Compile checked QuotientLift contributions directly into factored lanes.
/// The public stored-event builder and factor_common_alpha remain the oracle.
#[tracing::instrument(skip_all, name = "compile_relation_weight_factorization")]
pub(crate) fn compile_relation_weight_factorization<F, E>(
    inputs: RelationWeightEventInputs<'_, F, E>,
    projection: Option<&SetupColumnProjection<'_, F, E>>,
) -> Result<RelationWeightBuild<E, RelationWeightFactorization<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let (target, opening_semantics) =
        build_relation_weights(inputs, FactoredWeightSink::new, projection)?;
    Ok((target.finish()?, opening_semantics))
}

fn build_relation_weights<F, E, S>(
    inputs: RelationWeightEventInputs<'_, F, E>,
    target: impl FnOnce(Vec<E>, usize, usize, bool) -> Result<S, AkitaError>,
    projection: Option<&SetupColumnProjection<'_, F, E>>,
) -> Result<(S, OpeningFamily<(), CoefficientPackingBatchSemantics<E>>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
    S: RelationWeightTarget<E>,
{
    let RelationWeightEventInputs {
        setup,
        instance,
        alpha,
        level_params: lp,
        relation_row_point: tau1,
        claim_coefficients: gamma,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
        opening_points,
    } = inputs;
    let opening_batch = instance.opening_batch();
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let setup_matrix = match setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        RelationSetupSource::DeferredClaim => None,
    };
    let compilation = RelationWeightCompilation::new(
        setup_matrix,
        instance,
        lp,
        tau1,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
    )?;
    let role_dims = instance.role_dims();
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let packing_required = matches!(
        compilation.relation_geometry.group_opening_method(0)?,
        OpeningMethod::SubringCoefficientPacking { .. }
    );
    if packing_required != matches!(opening_points, OpeningFamily::SubringCoefficientPacking(_)) {
        return Err(AkitaError::InvalidSetup(
            "relation opening family disagrees with prepared points".into(),
        ));
    }
    let quotient_row_dims = compilation
        .row_families
        .iter()
        .map(|row| row.geometry().polynomial_modulus_dimension())
        .collect::<Vec<_>>();
    let rows = quotient_row_dims.len();
    if rows == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let mut additional_quotient_alpha_powers = Vec::new();
    for &row_dim in &quotient_row_dims {
        if row_dim != d_a
            && row_dim != d_b
            && row_dim != d_d
            && additional_quotient_alpha_powers
                .iter()
                .all(|(dimension, _): &(usize, Vec<E>)| *dimension != row_dim)
        {
            additional_quotient_alpha_powers.push((row_dim, scalar_powers(alpha, row_dim)));
        }
    }
    let levels = r_decomp_levels::<F>(lp.open().digits.log_basis);
    let setup_is_deferred = setup_matrix.is_none();
    let relation_coefficient_block_len = compilation.relation_coefficient_block_len;
    let physical_field_len = compilation.physical_field_len;
    let live_witness_coeff_len = compilation.witness_layout.live_coeff_len();
    let (coefficient_packing_events, opening_semantics) = match opening_points {
        OpeningFamily::SubringCoefficientPacking(prepared_points) => {
            let (events, batch) = prepare_coefficient_packing_batch_semantics(
                CoefficientPackingBatchSemanticInputs {
                    level_params: lp,
                    opening_batch,
                    relation_plan,
                    relation: instance,
                    prepared_points,
                    alpha,
                    tau1,
                    claim_coefficients: gamma,
                },
            )?;
            (events, OpeningFamily::SubringCoefficientPacking(batch))
        }
        OpeningFamily::EvaluationTrace(()) => (Vec::new(), OpeningFamily::EvaluationTrace(())),
    };
    let coefficient_packing_groups = match &opening_semantics {
        OpeningFamily::EvaluationTrace(()) => &[][..],
        OpeningFamily::SubringCoefficientPacking(batch) => batch.groups(),
    };
    let mut relation_events = target(
        scalar_powers(
            alpha,
            quotient_row_dims
                .iter()
                .copied()
                .max()
                .ok_or(AkitaError::InvalidProof)?,
        ),
        relation_coefficient_block_len,
        physical_field_len,
        setup_is_deferred,
    )?;
    let mut packing_semantics_by_group = vec![None; opening_batch.num_groups()];
    for semantics in coefficient_packing_groups {
        let group_index = semantics.group_index();
        let slot = packing_semantics_by_group
            .get_mut(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.replace(semantics).is_some() {
            return Err(AkitaError::InvalidSetup(
                "packing relation group appears more than once".into(),
            ));
        }
        if semantics.stage2_terms().physical_field_len() != live_witness_coeff_len {
            return Err(AkitaError::InvalidSetup(
                "packing relation live domain disagrees with the current ring switch".into(),
            ));
        }
        if semantics.stage2_terms().relation_coefficient_block_len()
            != relation_coefficient_block_len
        {
            return Err(AkitaError::InvalidSetup(
                "packing relation coefficient block disagrees with the current ring switch".into(),
            ));
        }
    }
    relation_events.extend_events(coefficient_packing_events)?;
    for group_plan in &compilation.plan.groups {
        let group_index = group_plan.group_index;
        let group_source = compilation.group_source(group_index)?;
        let group_setup = compilation
            .setup_sources
            .as_ref()
            .map(|sources| sources.group(group_index))
            .transpose()?;
        let packing_semantics = *packing_semantics_by_group
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let group_d_a = group_plan.roles.d_a;
        let group_d_b = group_plan.roles.d_b;
        let group_d_d = group_plan.roles.d_d;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let opening_method = group_plan.opening_method;
        match (opening_method, packing_semantics) {
            (OpeningMethod::EvaluationTrace, None) => {}
            (OpeningMethod::SubringCoefficientPacking { .. }, Some(semantics))
                if semantics.geometry().a_ring_dimension() == group_d_a => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "packing semantic groups do not match scheduled opening methods".into(),
                ));
            }
        }
        let ring_multiplier_point = match group_source.opening {
            OpeningFamily::EvaluationTrace(point) => Some(point),
            OpeningFamily::SubringCoefficientPacking(()) => None,
        };
        let challenges = group_source.challenges;
        let total_blocks = challenges.len();
        let challenge_evaluations = (0..total_blocks)
            .map(|index| challenges.eval_at_pows::<F, E>(index, &group_alpha_pows_a))
            .collect::<Result<Vec<_>, _>>()?;
        let (d_setup_accs, b_setup_accs, a_setup) =
            match (compilation.setup_sources.as_ref(), group_setup) {
                (Some(setup_sources), Some(group_setup)) => {
                    let _span =
                        tracing::info_span!("relation_weight_setup_columns_batch").entered();
                    let requests = [
                        ScalarSetupRequest {
                            family: &setup_sources.d,
                            columns: group_plan.rows.d_setup_range.clone(),
                            row_weights: &compilation.plan.d_row_weights,
                            batch_count: 1,
                            alpha_powers: &group_alpha_pows_d,
                        },
                        ScalarSetupRequest {
                            family: &group_setup.b,
                            columns: 0..group_plan.witness.b_width,
                            row_weights: &group_plan.rows.b_setup_row_weights,
                            batch_count: group_plan.witness.slice_count,
                            alpha_powers: &group_alpha_pows_b,
                        },
                        ScalarSetupRequest {
                            family: &group_setup.a,
                            columns: 0..group_plan.witness.inner_width,
                            row_weights: &group_plan.rows.a_setup_row_weights,
                            batch_count: 1,
                            alpha_powers: &group_alpha_pows_a,
                        },
                    ];
                    let matrix = setup_matrix.ok_or(AkitaError::InvalidProof)?;
                    let mut values =
                        contract_scalar_setup_batch(matrix, &requests, projection)?.into_iter();
                    (
                        Some(values.next().ok_or(AkitaError::InvalidProof)?),
                        Some(values.next().ok_or(AkitaError::InvalidProof)?),
                        Some(values.next().ok_or(AkitaError::InvalidProof)?),
                    )
                }
                (None, None) => (None, None, None),
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "lifted setup projection phases disagree".into(),
                    ))
                }
            };

        {
            let setup = match (d_setup_accs.as_ref(), b_setup_accs.as_ref()) {
                (Some(d), Some(b)) => LiftedEtSetup::Matrix { d, b },
                (None, None) => LiftedEtSetup::Deferred,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "lifted E/T setup phases disagree".into(),
                    ));
                }
            };
            let mut et_sink = LiftedEtSink {
                events: &mut relation_events,
                plan: group_plan,
                challenge_evaluations: &challenge_evaluations,
                setup,
            };
            compile_group_et_addresses(group_plan, &compilation.witness_layout, &mut et_sink)?;
        }
        // These setup-column accumulators can be large and are not used by
        // the z-hat phase below. Release them at the named phase boundary.
        drop(challenge_evaluations);
        drop(d_setup_accs);
        drop(b_setup_accs);

        // For z_hat[blk, dc, df], the column value is:
        //
        // -G_fold[df] * (
        //     tau_consistency * a_alpha[blk] * G_commit[dc]
        //     + sum_r tau_A[r] * A_alpha[r, blk, dc]
        //   ).
        //
        // The first term is the opening row. The second term is the A-row setup
        // contribution. A is already digit-domain, so the A-row setup term does
        // not multiply by G_commit.
        let opening_evaluations = if let Some(point) = ring_multiplier_point {
            (0..group_plan.witness.num_positions)
                .map(|position| point.eval_position_at::<E>(position, &group_alpha_pows_a))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![E::zero(); group_plan.witness.num_positions]
        };
        let setup = match a_setup.as_ref() {
            Some(values) => LiftedZSetup::Matrix(values),
            None => LiftedZSetup::Deferred,
        };
        let mut z_sink = LiftedZSink {
            events: &mut relation_events,
            plan: group_plan,
            opening_evaluations: &opening_evaluations,
            setup,
        };
        compile_group_z_addresses(group_plan, &compilation.witness_layout, &mut z_sink)?;
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.open().digits.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, &row_dim) in quotient_row_dims.iter().enumerate() {
        if matches!(
            compilation.row_families[row],
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        if matches!(
            compilation.row_families[row],
            RelationRowFamily::Consistency {
                opening_method: OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ) {
            continue;
        }
        let eq_weight = compilation.row_weights[row];
        let row_alpha_pows = if row_dim == d_a {
            relation_events.alpha_powers()
        } else if row_dim == d_b {
            alpha_pows_b.as_slice()
        } else if row_dim == d_d {
            alpha_pows_d.as_slice()
        } else {
            additional_quotient_alpha_powers
                .iter()
                .find_map(|(dimension, powers)| {
                    (*dimension == row_dim).then_some(powers.as_slice())
                })
                .ok_or(AkitaError::InvalidProof)?
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let physical_start = compilation
                .witness_layout
                .r_coefficient_index(row, digit, 0, 0)?;
            relation_events.push_native_ring(
                physical_start,
                row_dim,
                -(eq_weight * row_denom * *gadget),
                RelationWeightContribution::Constraint,
            )?;
        }
    }
    Ok((relation_events, opening_semantics))
}

#[cfg(test)]
#[path = "relation_weights_tests.rs"]
mod tests;
