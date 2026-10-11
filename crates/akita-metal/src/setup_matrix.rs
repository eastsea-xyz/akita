//! Exact, bounded public setup expansion on the GPU.

use std::mem::size_of;
use std::time::Duration;

use akita_types::{derive_public_matrix_prefix, AkitaSetupSeed, FlatMatrix};
use jolt_field::Prime128OffsetA7F7;

use crate::field::F;
use crate::runtime::setup_matrix::{
    SETUP_MATRIX_DRAW_LIMIT, SETUP_MATRIX_FIELDS_PER_PAGE, SETUP_MATRIX_MAX_PAGES,
};
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

#[cfg(test)]
mod tests;

/// Work and timings observed while expanding a public setup prefix.
#[derive(Clone, Debug, Default)]
pub struct MetalSetupMatrixMetrics {
    /// Requested canonical field bytes, excluding temporary device storage.
    pub output_bytes: usize,
    /// Requested independently derived pages, including a partial final page.
    pub pages: usize,
    /// Field elements returned by completed GPU commands.
    pub gpu_field_elements: usize,
    /// Successfully completed GPU commands.
    pub command_buffers: usize,
    /// Largest per-command device allocation, including input and status buffers.
    pub peak_allocation_bytes: usize,
    /// Time allocating and binding per-command buffers.
    pub buffer_setup_time: Duration,
    /// Wall time committing and waiting for completed GPU commands.
    pub command_wall_time: Duration,
    /// Sum of GPU timestamp intervals when the device exposes them.
    pub gpu_active_time: Duration,
    /// Time copying and checking canonical GPU output.
    pub readback_time: Duration,
    /// Whether explicitly preferred Metal delegated this operation to CPU.
    pub cpu_fallback: bool,
    /// Original device/admission error when preferred Metal used CPU.
    pub fallback_reason: Option<String>,
}

/// Exact canonical matrix together with evidence of the selected compute route.
#[derive(Debug)]
pub struct MetalSetupMatrixOutput {
    /// Flat canonical prefix, independent of subsequent ring-matrix views.
    pub matrix: FlatMatrix<Prime128OffsetA7F7>,
    /// Bounded allocation and completed GPU work measurements.
    pub metrics: MetalSetupMatrixMetrics,
}

impl MetalBackend {
    /// Expand the canonical paged SHAKE256 setup for the production 128-bit field.
    ///
    /// Device output is bounded to 256 pages per command. Required Metal errors
    /// propagate, so a caller can retry its setup/proof on CPU. Only the explicit
    /// `PreferMetal` policy delegates an operation here to canonical CPU expansion.
    pub fn derive_public_matrix_prefix(
        &self,
        num_field_elements: usize,
        seed: &AkitaSetupSeed,
    ) -> Result<MetalSetupMatrixOutput, MetalCommitError> {
        self.derive_setup_matrix_with_limit(num_field_elements, seed, SETUP_MATRIX_DRAW_LIMIT)
    }

    fn derive_setup_matrix_with_limit(
        &self,
        num_field_elements: usize,
        seed: &AkitaSetupSeed,
        draw_limit: u32,
    ) -> Result<MetalSetupMatrixOutput, MetalCommitError> {
        let output_bytes = num_field_elements
            .checked_mul(size_of::<F>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(MetalCommitError::ShapeOverflow("setup matrix field bytes"))?;
        let mut metrics = MetalSetupMatrixMetrics {
            output_bytes,
            pages: num_field_elements.div_ceil(SETUP_MATRIX_FIELDS_PER_PAGE),
            ..Default::default()
        };
        if num_field_elements == 0 {
            return Ok(MetalSetupMatrixOutput {
                matrix: FlatMatrix::from_flat_data(Vec::new()),
                metrics,
            });
        }
        let attempt: Result<FlatMatrix<F>, MetalCommitError> = (|| {
            let runtime = self.runtime().ok_or(MetalCommitError::DeviceUnavailable)?;
            let mut fields = Vec::new();
            fields
                .try_reserve_exact(num_field_elements)
                .map_err(|error| {
                    MetalCommitError::UnsupportedShape(format!(
                        "setup matrix host allocation: {error}"
                    ))
                })?;
            let window_fields = SETUP_MATRIX_MAX_PAGES * SETUP_MATRIX_FIELDS_PER_PAGE;
            while fields.len() < num_field_elements {
                let first_page = u64::try_from(fields.len() / SETUP_MATRIX_FIELDS_PER_PAGE)
                    .map_err(|_| MetalCommitError::ShapeOverflow("setup matrix first page"))?;
                let count = (num_field_elements - fields.len()).min(window_fields);
                let mut outcome = runtime
                    .dispatch_setup_matrix_with_limit(seed, first_page, count, draw_limit)?;
                metrics.gpu_field_elements += outcome.fields.len();
                metrics.command_buffers += 1;
                metrics.peak_allocation_bytes =
                    metrics.peak_allocation_bytes.max(outcome.allocation_bytes);
                metrics.buffer_setup_time += outcome.timings.buffer_setup;
                metrics.command_wall_time += outcome.timings.command_wall;
                metrics.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
                metrics.readback_time += outcome.timings.readback_copy;
                fields.append(&mut outcome.fields);
            }
            Ok(FlatMatrix::from_flat_data(fields))
        })();
        let matrix = match attempt {
            Ok(matrix) => matrix,
            Err(error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                // The failed attempt has released its partial matrix and every
                // completed device window before CPU allocates the exact prefix.
                metrics.cpu_fallback = true;
                metrics.fallback_reason = Some(error.to_string());
                tracing::warn!(%error, "preferred Metal setup expansion is using CPU");
                derive_public_matrix_prefix::<F>(num_field_elements, seed)
            }
            Err(error) => return Err(error),
        };
        tracing::info!(
            fields = num_field_elements,
            pages = metrics.pages,
            commands = metrics.command_buffers,
            gpu_fields = metrics.gpu_field_elements,
            gpu_seconds = metrics.gpu_active_time.as_secs_f64(),
            cpu_fallback = metrics.cpu_fallback,
            "public setup matrix expansion completed"
        );
        Ok(MetalSetupMatrixOutput { matrix, metrics })
    }
}
