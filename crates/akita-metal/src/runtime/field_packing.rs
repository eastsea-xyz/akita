use super::*;

/// Canonical extension coordinates are contiguous within each output block.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FieldCoefficientPackingParams {
    pub(crate) live_coefficients: u64,
    pub(crate) num_live_positions: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) ring_d: u64,
    pub(crate) stride: u64,
    pub(crate) subring_dimension: u64,
    pub(crate) extension_degree: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) bit_width: u64,
}

pub(crate) enum FieldPackingSource<'a> {
    Field {
        buffer: crate::prepared::MatrixBuffer,
        coefficients: usize,
    },
    Packed {
        bytes: &'a [u8],
        bit_width: u8,
        stored_coefficients: usize,
        live_coefficients: usize,
    },
}

enum FieldPackingBuffer<'a> {
    Field(crate::prepared::MatrixBuffer),
    Packed(SharedSliceBuffer<'a, u8>),
}

impl std::ops::Deref for FieldPackingBuffer<'_> {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        match self {
            Self::Field(buffer) => buffer,
            Self::Packed(shared) => &shared.buffer,
        }
    }
}

impl MetalRuntime {
    pub(crate) fn admit_field_coefficient_packing(
        &self,
        params: FieldCoefficientPackingParams,
        weight_count: usize,
        source_count: usize,
    ) -> Result<u64, MetalCommitError> {
        let weights = params
            .positions_per_block
            .checked_mul(params.stride)
            .and_then(|count| count.checked_mul(params.extension_degree))
            .ok_or(MetalCommitError::ShapeOverflow("field packing weights"))?;
        let output = params
            .num_blocks
            .checked_mul(params.subring_dimension)
            .and_then(|count| count.checked_mul(params.extension_degree))
            .ok_or(MetalCommitError::ShapeOverflow("field packing output"))?;
        params.num_live_positions.checked_mul(params.ring_d).ok_or(
            MetalCommitError::ShapeOverflow("field packing source extent"),
        )?;
        if source_count == 0
            || params.ring_d == 0
            || params.stride == 0
            || params.subring_dimension == 0
            || !matches!(params.extension_degree, 1 | 2)
            || params.stride.checked_mul(params.subring_dimension) != Some(params.ring_d)
            || params.num_live_positions == 0
            || params.positions_per_block == 0
            || params.num_blocks
                != params
                    .num_live_positions
                    .div_ceil(params.positions_per_block)
            || params.output_coefficients != output
            || output > u64::from(u32::MAX)
            || u64::try_from(weight_count).ok() != Some(weights)
            || self
                .fp128_field_coefficient_packing_pipeline
                .max_total_threads_per_threadgroup()
                < FP128_COEFFICIENT_PACKING_THREADS as u64
            || self
                .fp128_packed_digit_coefficient_packing_pipeline
                .max_total_threads_per_threadgroup()
                < FP128_COEFFICIENT_PACKING_THREADS as u64
        {
            return Err(MetalCommitError::UnsupportedShape(
                "fp128 field packing geometry is unsupported".into(),
            ));
        }
        self.validate_buffer_length(weight_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("field packing weight bytes"),
        )?)?;
        let batch_output = usize::try_from(output)
            .ok()
            .and_then(|n| n.checked_mul(source_count))
            .and_then(|n| n.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "field packing batch output bytes",
            ))?;
        self.validate_buffer_length(batch_output)?;
        Ok(output)
    }

    pub(crate) fn admit_field_packing_source_bytes(
        &self,
        coefficients: usize,
    ) -> Result<(), MetalCommitError> {
        self.validate_buffer_length(
            coefficients
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("field packing input bytes"))?,
        )
    }

    pub(crate) fn dispatch_fp128_field_coefficient_packing(
        &self,
        sources: &[FieldPackingSource<'_>],
        combined_weights: &[Fp128Limbs],
        params: FieldCoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let output = self.admit_field_coefficient_packing(
                params,
                combined_weights.len(),
                sources.len(),
            )?;
            let live_capacity = params.num_live_positions.checked_mul(params.ring_d).ok_or(
                MetalCommitError::ShapeOverflow("field packing source extent"),
            )?;
            // Admit the complete batch before uploading or submitting any work.
            let mut source_params = Vec::with_capacity(sources.len());
            for source in sources {
                let (coefficients, bit_width) = match source {
                    FieldPackingSource::Field {
                        buffer,
                        coefficients,
                    } => {
                        let bytes = coefficients
                            .checked_mul(size_of::<Fp128Limbs>())
                            .ok_or(MetalCommitError::ShapeOverflow("field packing input bytes"))?;
                        self.admit_field_packing_source_bytes(*coefficients)?;
                        if usize::try_from(buffer.length())
                            .ok()
                            .is_none_or(|len| len < bytes)
                        {
                            return Err(MetalCommitError::UnsupportedShape(
                                "field packing input buffer is undersized".into(),
                            ));
                        }
                        (*coefficients, 0)
                    }
                    FieldPackingSource::Packed {
                        bytes,
                        bit_width,
                        stored_coefficients,
                        live_coefficients,
                    } => {
                        let required = stored_coefficients
                            .checked_mul(usize::from(*bit_width))
                            .map(|bits| bits.div_ceil(8))
                            .ok_or(MetalCommitError::ShapeOverflow(
                                "packed packing input bytes",
                            ))?;
                        self.validate_buffer_length(bytes.len())?;
                        if !(1..=8).contains(bit_width)
                            || bytes.len() != required
                            || live_coefficients > stored_coefficients
                        {
                            return Err(MetalCommitError::UnsupportedShape(
                                "packed packing input extent is invalid".into(),
                            ));
                        }
                        (*live_coefficients, u64::from(*bit_width))
                    }
                };
                let coefficients = u64::try_from(coefficients).map_err(|_| {
                    MetalCommitError::ShapeOverflow("field packing live coefficients")
                })?;
                if coefficients == 0
                    || coefficients > live_capacity
                    || coefficients.div_ceil(params.ring_d) != params.num_live_positions
                {
                    return Err(MetalCommitError::UnsupportedShape(
                        "field packing live positions disagree with source".into(),
                    ));
                }
                source_params.push(FieldCoefficientPackingParams {
                    live_coefficients: coefficients,
                    bit_width,
                    ..params
                });
            }
            let buffer_start = Instant::now();
            let weights_buffer = self.shared_buffer_from_slice(combined_weights)?;
            let mut inputs = Vec::with_capacity(sources.len());
            let mut input_bytes = 0usize;
            for source in sources {
                let input = match source {
                    FieldPackingSource::Field { buffer, .. } => {
                        FieldPackingBuffer::Field(buffer.clone())
                    }
                    FieldPackingSource::Packed { bytes, .. } => {
                        let buffer = self.shared_slice_buffer(bytes)?;
                        if !buffer.zero_copy {
                            input_bytes = input_bytes.checked_add(bytes.len()).ok_or(
                                MetalCommitError::ShapeOverflow("field packing input allocation"),
                            )?;
                        }
                        FieldPackingBuffer::Packed(buffer)
                    }
                };
                inputs.push(input);
            }
            let per_source_count = usize::try_from(output)
                .map_err(|_| MetalCommitError::ShapeOverflow("field packing output count"))?;
            let output_count = per_source_count.checked_mul(sources.len()).ok_or(
                MetalCommitError::ShapeOverflow("field packing batch output"),
            )?;
            let source_output_bytes = per_source_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "field packing output bytes",
                ))?;
            let output_bytes = source_output_bytes.checked_mul(sources.len()).ok_or(
                MetalCommitError::ShapeOverflow("field packing batch output bytes"),
            )?;
            let output_buffer = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita resident field coefficient packing batch");
            let encoder = command.new_compute_command_encoder();
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            for (index, (input, params)) in inputs.iter().zip(&source_params).enumerate() {
                encoder.set_compute_pipeline_state(if params.bit_width == 0 {
                    &self.fp128_field_coefficient_packing_pipeline
                } else {
                    &self.fp128_packed_digit_coefficient_packing_pipeline
                });
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(
                    2,
                    Some(&output_buffer),
                    (index * source_output_bytes) as u64,
                );
                set_inline_bytes(encoder, 3, params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(output, 1, 1),
                    MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
                );
            }
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: completed dispatches wrote every canonical limb in disjoint
            // aligned source output ranges; all borrowed inputs remain alive.
            let coefficients = unsafe {
                std::slice::from_raw_parts(
                    output_buffer.contents().cast::<Fp128Limbs>(),
                    output_count,
                )
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
                allocation_bytes: output_bytes
                    .checked_add(size_of_val(combined_weights))
                    .and_then(|bytes| bytes.checked_add(input_bytes))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "field packing allocation bytes",
                    ))?,
            })
        })
    }
}
