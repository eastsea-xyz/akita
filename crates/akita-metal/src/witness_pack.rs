use akita_error::AkitaError;
use akita_prover::compute::PackedWitnessZ;

use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

impl MetalBackend {
    pub(crate) fn pack_centered_witness_z<const D: usize>(
        &self,
        centered: &[i32],
        fold_planes: usize,
        log_basis: u32,
        bit_width: u8,
    ) -> Result<Option<PackedWitnessZ>, AkitaError> {
        // Admission is decided before execution. Once submitted, failures remain errors.
        if self.policy() == MetalExecutionPolicy::PreferMetal
            && centered
                .len()
                .checked_mul(fold_planes)
                .is_some_and(|count| count < (1 << 16))
        {
            return Ok(None);
        }
        let Some(runtime) = self.runtime() else {
            return if self.policy() == MetalExecutionPolicy::PreferMetal {
                Ok(None)
            } else {
                Err(MetalCommitError::DeviceUnavailable.into_akita())
            };
        };
        let output = runtime
            .pack_centered_witness_z::<D>(centered, fold_planes, log_basis, bit_width)
            .map_err(MetalCommitError::into_akita)?;
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += output.timings.command_wall;
            metrics.gpu_active_time += output.timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += output.timings.buffer_setup;
            metrics.readback_time += output.timings.readback_copy;
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(Some(PackedWitnessZ {
            bytes: output.bytes,
            digit_len: output.digit_len,
            negative_abs_max: output.negative_abs_max,
            positive_max: output.positive_max,
        }))
    }
}
