use super::*;
use crate::prepared::MatrixBuffer;
use jolt_field::Zero;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SetupContractionParams {
    pub(crate) ring_d: u64,
    pub(crate) matrix_columns: u64,
    pub(crate) first_column: u64,
    pub(crate) column_count: u64,
    pub(crate) selected_rows: u64,
    pub(crate) batch_count: u64,
    pub(crate) output_count: u64,
}

pub(crate) struct SetupContractionInput {
    pub(crate) params: SetupContractionParams,
    pub(crate) row_ids: Vec<u64>,
    pub(crate) row_weights: Vec<F>,
    pub(crate) alpha_powers: Vec<F>,
}

impl MetalRuntime {
    pub(crate) fn admit_setup_contractions(
        &self,
        matrix_fields: usize,
        inputs: &[SetupContractionInput],
    ) -> Result<usize, MetalCommitError> {
        let mut outputs = 0usize;
        for input in inputs {
            let p = input.params;
            let count = p
                .column_count
                .checked_mul(p.batch_count)
                .ok_or(MetalCommitError::ShapeOverflow("setup contraction output"))?;
            let end_column = p.first_column.checked_add(p.column_count).ok_or(
                MetalCommitError::ShapeOverflow("setup contraction column extent"),
            )?;
            let weights = p
                .selected_rows
                .checked_mul(p.batch_count)
                .ok_or(MetalCommitError::ShapeOverflow("setup contraction weights"))?;
            p.selected_rows
                .checked_mul(p.ring_d)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "setup contraction reduction terms",
                ))?;
            if p.ring_d == 0
                || p.batch_count == 0
                || end_column > p.matrix_columns
                || p.output_count != count
                || count > u64::from(u32::MAX)
                || input.row_ids.len() as u64 != p.selected_rows
                || input.row_weights.len() as u64 != weights
                || input.alpha_powers.len() as u64 != p.ring_d
                || self
                    .setup_contraction_pipeline
                    .max_total_threads_per_threadgroup()
                    < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "scalar setup contraction geometry".into(),
                ));
            }
            for &row in &input.row_ids {
                let fields = row
                    .checked_add(1)
                    .and_then(|rows| rows.checked_mul(p.matrix_columns))
                    .and_then(|n| n.checked_mul(p.ring_d))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "setup contraction matrix extent",
                    ))?;
                if fields > matrix_fields as u64 {
                    return Err(MetalCommitError::UnsupportedShape(
                        "setup contraction matrix address".into(),
                    ));
                }
            }
            self.validate_buffer_length(
                size_of_val(input.row_ids.as_slice()).max(size_of::<u64>()),
            )?;
            self.validate_buffer_length(
                size_of_val(input.row_weights.as_slice()).max(size_of::<F>()),
            )?;
            self.validate_buffer_length(size_of_val(input.alpha_powers.as_slice()))?;
            outputs =
                outputs
                    .checked_add(count as usize)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "setup contraction batch outputs",
                    ))?;
        }
        let bytes =
            outputs
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "setup contraction batch output bytes",
                ))?;
        self.validate_buffer_length(bytes)?;
        Ok(outputs)
    }

    pub(crate) fn dispatch_setup_contractions(
        &self,
        matrix: &MatrixBuffer,
        matrix_fields: usize,
        inputs: &[SetupContractionInput],
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let output_count = self.admit_setup_contractions(matrix_fields, inputs)?;
            let matrix_bytes = matrix_fields.checked_mul(size_of::<F>()).ok_or(
                MetalCommitError::ShapeOverflow("setup contraction matrix bytes"),
            )?;
            if matrix.length() < matrix_bytes as u64 {
                return Err(MetalCommitError::UnsupportedShape(
                    "setup contraction matrix buffer extent".into(),
                ));
            }
            if output_count == 0 {
                return Ok(CoefficientPackingDispatchOutcome {
                    coefficients: Vec::new(),
                    timings: DispatchTimings::default(),
                    allocation_bytes: 0,
                });
            }
            let buffer_start = Instant::now();
            let output_bytes = output_count * size_of::<Fp128Limbs>();
            let output = self.shared_buffer(output_bytes)?;
            let mut weights = Vec::with_capacity(inputs.len());
            let mut rows = Vec::with_capacity(inputs.len());
            let mut powers = Vec::with_capacity(inputs.len());
            let mut allocation_bytes = output_bytes;
            let zero = [F::zero()];
            let zero_row = [0u64];
            for input in inputs {
                let row_values = if input.row_ids.is_empty() {
                    &zero_row[..]
                } else {
                    &input.row_ids
                };
                let weight_values = if input.row_weights.is_empty() {
                    &zero[..]
                } else {
                    &input.row_weights
                };
                rows.push(self.shared_buffer_from_slice(row_values)?);
                weights.push(self.shared_buffer_from_slice(weight_values)?);
                powers.push(self.shared_buffer_from_slice(&input.alpha_powers)?);
                allocation_bytes = allocation_bytes
                    .checked_add(size_of_val(row_values))
                    .and_then(|n| n.checked_add(size_of_val(weight_values)))
                    .and_then(|n| n.checked_add(size_of_val(input.alpha_powers.as_slice())))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "setup contraction allocation",
                    ))?;
            }
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita scalar setup column contractions");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.setup_contraction_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            let mut offset = 0u64;
            for (index, input) in inputs.iter().enumerate() {
                if input.params.output_count == 0 {
                    continue;
                }
                encoder.set_buffer(1, Some(&rows[index]), 0);
                encoder.set_buffer(2, Some(&weights[index]), 0);
                encoder.set_buffer(3, Some(&powers[index]), 0);
                encoder.set_buffer(4, Some(&output), offset);
                set_inline_bytes(encoder, 5, &input.params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(input.params.output_count, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                offset += input.params.output_count * size_of::<Fp128Limbs>() as u64;
            }
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: completed disjoint dispatches wrote all output canonical limbs.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes,
            })
        })
    }
}
