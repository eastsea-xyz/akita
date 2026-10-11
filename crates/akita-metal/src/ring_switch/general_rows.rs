use super::*;
use crate::runtime::{RelationRowsInput, RelationRowsMode, RelationRowsParams};
use akita_error::checked;

impl MetalBackend {
    pub(super) fn general_relation_rows<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<Option<RingSwitchRelationRows<F, D>>, AkitaError> {
        if !matches!(D, 64 | 128 | 256 | 512) {
            return Ok(None);
        }
        let Some(runtime) = self.runtime() else {
            return Ok(None);
        };
        if (plan.n_b != 0 || plan.n_a != 0)
            && !(1..=akita_types::MAX_I8_LOG_BASIS).contains(&plan.log_basis_outer)
        {
            return Err(AkitaError::InvalidInput(
                "invalid outer relation digit basis".into(),
            ));
        }
        let input_d = (plan.n_d != 0)
            .then(|| RelationRowsInput::from_i8(source.e_hat, plan.log_basis_open))
            .transpose()
            .map_err(MetalCommitError::into_akita)?;
        let input_b = (plan.n_b != 0)
            .then(|| RelationRowsInput::from_i8(source.t_hat, plan.log_basis_outer))
            .transpose()
            .map_err(MetalCommitError::into_akita)?;
        let input_a = (plan.n_a != 0).then(|| {
            RelationRowsInput::from_i32(source.z_segment, source.z_folded_centered_inf_norm)
        });
        let roles = [
            (plan.n_d, input_d.as_ref(), RelationRowsMode::Paired),
            (plan.n_b, input_b.as_ref(), RelationRowsMode::Cyclic),
            (plan.n_a, input_a.as_ref(), RelationRowsMode::High),
        ];
        let mut extent = 0usize;
        for (rows, input, _) in &roles {
            let Some(input) = input else {
                continue;
            };
            if input.len() == 0 {
                return Err(AkitaError::InvalidInput(
                    "active relation role has no source columns".into(),
                ));
            }
            extent = extent.max(checked::product([*rows, input.len()]).ok_or_else(|| {
                MetalCommitError::ShapeOverflow("relation matrix extent").into_akita()
            })?);
        }
        if extent == 0 {
            return Ok(None);
        }
        // Check every active role, including all allocation/grid and exact CRT
        // limits, before transforming any matrix or submitting any command.
        for (rows, input, mode) in &roles {
            let Some(input) = input else {
                continue;
            };
            let params = RelationRowsParams::new(*rows, extent, *mode, input)
                .map_err(MetalCommitError::into_akita)?;
            if !runtime.supports_fp128_relation_rows::<D>(params) {
                return Ok(None);
            }
        }
        let matrix = prepared.linear_matrix::<D>(runtime, extent)?;
        let stride_bytes = checked::product([
            2,
            D,
            crate::runtime::FP128_D512_LINEAR_RELATION_NUM_PRIMES,
            size_of::<i32>(),
        ])
        .ok_or_else(|| MetalCommitError::ShapeOverflow("relation spectrum stride").into_akita())?;
        if !matrix.bytes.is_multiple_of(stride_bytes) {
            return Err(AkitaError::InvalidProof);
        }
        let matrix_rings = matrix.bytes / stride_bytes;
        let mut output = RingSwitchRelationRows {
            d_negacyclic: Vec::new(),
            d_cyclic: Vec::new(),
            b_cyclic: Vec::new(),
            a_quotients: Vec::new(),
        };
        self.update_opening_metrics(|metrics| {
            metrics.buffer_setup_time += matrix.prepare_time;
            if !matrix.cache_hit {
                metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(matrix.bytes);
            }
        })
        .map_err(MetalCommitError::into_akita)?;
        for (role, (rows, input, mode)) in roles.iter().enumerate() {
            let Some(input) = input else {
                continue;
            };
            let params = RelationRowsParams::new(*rows, matrix_rings, *mode, input)
                .map_err(MetalCommitError::into_akita)?;
            let result = runtime
                .dispatch_fp128_relation_rows(&matrix.buffer, input, params)
                .map_err(MetalCommitError::into_akita)?;
            let coefficients = result
                .coefficients
                .into_iter()
                .enumerate()
                .map(|(index, value)| F::from_device(value, index))
                .collect::<Result<Vec<_>, _>>()
                .map_err(MetalCommitError::into_akita)?;
            let timings = result.timings;
            self.update_opening_metrics(|metrics| {
                metrics.command_wall_time += timings.command_wall;
                metrics.gpu_active_time += timings.gpu.unwrap_or_default();
                metrics.buffer_setup_time += timings.buffer_setup;
                metrics.readback_time += timings.readback_copy;
                metrics.allocation_bytes = metrics
                    .allocation_bytes
                    .saturating_add(result.allocation_bytes);
            })
            .map_err(MetalCommitError::into_akita)?;
            let rings = |values: &[F]| {
                values
                    .chunks_exact(D)
                    .map(CyclotomicRing::from_slice)
                    .collect::<Vec<_>>()
            };
            match role {
                0 => {
                    let (negacyclic, cyclic) = coefficients.split_at(rows * D);
                    output.d_negacyclic = rings(negacyclic);
                    output.d_cyclic = rings(cyclic);
                }
                1 => output.b_cyclic = rings(&coefficients),
                2 => output.a_quotients = rings(&coefficients),
                _ => return Err(AkitaError::InvalidProof),
            }
        }
        tracing::debug!(
            route = "metal-general",
            ring_dimension = D,
            n_d = plan.n_d,
            n_b = plan.n_b,
            n_a = plan.n_a,
            "completed general Metal relation rows"
        );
        Ok(Some(output))
    }
}
