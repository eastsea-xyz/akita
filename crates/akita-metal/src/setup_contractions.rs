use crate::field::{MetalField, F};
use crate::prepared::MatrixBuffer;
use crate::runtime::{SetupContractionInput, SetupContractionParams};
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy, MetalPreparedSetup};
use akita_error::{checked, AkitaError};
use akita_prover::compute::SetupColumnContractionPlan;
use akita_types::FpExtEncoding;
use jolt_field::Zero;

#[cfg(test)]
mod tests;

impl MetalBackend {
    pub(crate) fn scalar_setup_contractions<E: FpExtEncoding<F>>(
        &self,
        prepared: &MetalPreparedSetup,
        plans: &[SetupColumnContractionPlan<'_, E>],
    ) -> Result<Option<Vec<Vec<E>>>, AkitaError> {
        if E::DEGREE != 1 {
            return Ok(None);
        }
        let fields = prepared.expanded.shared_matrix().as_field_slice().len();
        let mut inputs = Vec::with_capacity(plans.len());
        let mut counts = Vec::with_capacity(plans.len());
        let mut largest = None;
        let mut largest_fields = 0usize;
        for plan in plans {
            let count = plan.validate(fields)?;
            let required = checked::product([plan.matrix_rows, plan.matrix_columns, plan.ring_d])
                .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("setup contraction matrix prefix").into_akita()
            })?;
            if required > largest_fields {
                largest_fields = required;
                largest = Some(plan);
            }
            let weights = plan
                .row_weights
                .iter()
                .flat_map(|(_, weights)| weights.iter())
                .map(|value| {
                    value.degree_one_base().ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "setup contraction scalar weight has no base coordinate".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let powers = plan
                .alpha_powers
                .iter()
                .map(|value| {
                    value.degree_one_base().ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "setup contraction alpha power has no base coordinate".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            inputs.push(SetupContractionInput {
                params: SetupContractionParams {
                    ring_d: plan.ring_d as u64,
                    matrix_columns: plan.matrix_columns as u64,
                    first_column: plan.columns.start as u64,
                    column_count: plan.columns.len() as u64,
                    selected_rows: plan.row_weights.len() as u64,
                    batch_count: plan.batch_count as u64,
                    output_count: count as u64,
                },
                row_ids: plan
                    .row_weights
                    .iter()
                    .map(|(row, _)| *row as u64)
                    .collect(),
                row_weights: weights,
                alpha_powers: powers,
            });
            counts.push(count);
        }
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::PreferMetal => Ok(None),
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
            };
        };
        // Admit all requests before preparing even the first canonical matrix prefix.
        match runtime.admit_setup_contractions(fields, &inputs) {
            Ok(_) => {}
            Err(MetalCommitError::UnsupportedShape(_) | MetalCommitError::BufferTooLong { .. })
                if self.policy() == MetalExecutionPolicy::PreferMetal =>
            {
                return Ok(None)
            }
            Err(error) => return Err(error.into_akita()),
        }
        if counts.iter().all(|&count| count == 0) {
            return Ok(Some(counts.iter().map(|_| Vec::new()).collect()));
        }
        let matrix = if let Some(plan) = largest {
            let matrix =
                prepared.matrix(runtime, plan.ring_d, plan.matrix_rows, plan.matrix_columns)?;
            self.update_opening_metrics(|m| {
                m.buffer_setup_time += matrix.prepare_time;
                if !matrix.cache_hit {
                    m.allocation_bytes = m
                        .allocation_bytes
                        .saturating_add(matrix.buffer.allocation_bytes());
                }
            })
            .map_err(MetalCommitError::into_akita)?;
            matrix.buffer.as_ref().clone()
        } else {
            // No rows are selected; the GPU writes canonical zero results.
            MatrixBuffer::from(
                runtime
                    .shared_buffer_from_slice(&[F::zero()])
                    .map_err(MetalCommitError::into_akita)?,
            )
        };
        let outcome = runtime
            .dispatch_setup_contractions(&matrix, largest_fields, &inputs)
            .map_err(MetalCommitError::into_akita)?;
        self.update_opening_metrics(|m| {
            m.buffer_setup_time += outcome.timings.buffer_setup;
            m.command_wall_time += outcome.timings.command_wall;
            m.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
            m.readback_time += outcome.timings.readback_copy;
            m.allocation_bytes = m.allocation_bytes.saturating_add(outcome.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        let mut coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index).map(E::lift_base))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?
            .into_iter();
        Ok(Some(
            counts
                .into_iter()
                .map(|count| coefficients.by_ref().take(count).collect())
                .collect(),
        ))
    }
}
