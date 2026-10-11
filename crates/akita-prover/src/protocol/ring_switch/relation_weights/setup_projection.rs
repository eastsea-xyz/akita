use super::setup_columns::{contract_setup_columns, SetupColumnValues, SetupRows};
use crate::compute::SetupColumnContractionPlan;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{Field, MulBaseUnreduced};
use std::ops::Range;

pub(crate) type SetupColumnProjection<'a, F, E> = dyn Fn(
        &AkitaExpandedSetup<F>,
        &[SetupColumnContractionPlan<'_, E>],
    ) -> Result<Option<Vec<Vec<E>>>, AkitaError>
    + Sync
    + 'a;

pub(super) struct ScalarSetupRequest<'a, F: Field, E: Field> {
    pub(super) family: &'a SetupRows<'a, F>,
    pub(super) columns: Range<usize>,
    pub(super) row_weights: &'a [(usize, Vec<E>)],
    pub(super) batch_count: usize,
    pub(super) alpha_powers: &'a [E],
}

pub(super) fn contract_scalar_setup_batch<F, E>(
    setup: &AkitaExpandedSetup<F>,
    requests: &[ScalarSetupRequest<'_, F, E>],
    projection: Option<&SetupColumnProjection<'_, F, E>>,
) -> Result<Vec<SetupColumnValues<E>>, AkitaError>
where
    F: Field,
    E: Field + MulBaseUnreduced<F>,
{
    if let Some(project) = projection {
        let plans = requests
            .iter()
            .map(|request| {
                let width = request.family.rows.first().map_or(0, |row| row.len());
                if request.family.ring_d == 0
                    || !width.is_multiple_of(request.family.ring_d)
                    || request.family.rows.iter().any(|row| row.len() != width)
                {
                    return Err(AkitaError::InvalidSetup(
                        "setup projection rows have inconsistent physical widths".into(),
                    ));
                }
                Ok(SetupColumnContractionPlan {
                    matrix_rows: request.family.rows.len(),
                    matrix_columns: width / request.family.ring_d,
                    ring_d: request.family.ring_d,
                    columns: request.columns.clone(),
                    row_weights: request.row_weights,
                    alpha_powers: request.alpha_powers,
                    batch_count: request.batch_count,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for plan in &plans {
            plan.validate(setup.shared_matrix().as_field_slice().len())?;
        }
        if let Some(outputs) = project(setup, &plans)? {
            if outputs.len() != requests.len() {
                return Err(AkitaError::InvalidSetup(
                    "setup projection request count mismatch".into(),
                ));
            }
            return outputs
                .into_iter()
                .zip(&plans)
                .map(|(values, plan)| {
                    SetupColumnValues::from_scalar_values(
                        plan.batch_count,
                        plan.columns.len(),
                        values,
                    )
                })
                .collect();
        }
    }
    requests
        .iter()
        .map(|request| {
            contract_setup_columns(
                request.family,
                request.columns.clone(),
                request.row_weights,
                request.batch_count,
                1,
                |coefficients| {
                    Ok([eval_flat_ring_at_pows_fast(
                        coefficients,
                        request.alpha_powers,
                    )])
                },
            )
        })
        .collect()
}
