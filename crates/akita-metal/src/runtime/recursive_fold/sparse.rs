use super::*;

impl MetalRuntime {
    pub(crate) fn admit_recursive_sparse_plans<'a>(
        &self,
        plans: impl IntoIterator<Item = (&'a RecursiveFoldParams, &'a [RecursiveSparseTerm])>,
    ) -> Result<(), MetalCommitError> {
        let plans = plans.into_iter().collect::<Vec<_>>();
        let first = *plans
            .first()
            .ok_or_else(|| {
                MetalCommitError::UnsupportedShape("empty recursive sparse plans".into())
            })?
            .0;
        let output_count = first
            .positions_per_block
            .checked_mul(first.num_digits)
            .and_then(|n| n.checked_mul(first.ring_d))
            .ok_or(MetalCommitError::ShapeOverflow("recursive sparse output"))?;
        let output_bytes = usize::try_from(output_count)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or(MetalCommitError::ShapeOverflow(
                "recursive sparse output bytes",
            ))?;
        let partial_bytes =
            output_bytes
                .checked_mul(plans.len())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive sparse partial bytes",
                ))?;
        self.validate_buffer_length(output_bytes)?;
        self.validate_buffer_length(partial_bytes)?;
        let mut absolute_bound = 0u64;
        for (p, terms) in plans {
            self.validate_recursive_fold_params(*p)?;
            if p.positions_per_block != first.positions_per_block
                || p.num_digits != first.num_digits
                || p.ring_d != first.ring_d
                || output_count > u64::from(u32::MAX)
                || terms
                    .iter()
                    .any(|t| t.block >= p.num_blocks || u64::from(t.shift) >= p.ring_d)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "recursive sparse batch geometry".into(),
                ));
            }
            let bits = if p.bit_width == 0 {
                p.log_basis
            } else {
                p.bit_width
            };
            let digit_bound = 1u64 << (bits - 1);
            let mass = terms
                .iter()
                .try_fold(0u64, |sum, t| {
                    sum.checked_add(u64::from(t.coefficient.unsigned_abs()))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive sparse challenge mass",
                ))?;
            absolute_bound = digit_bound
                .checked_mul(mass)
                .and_then(|n| absolute_bound.checked_add(n))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive sparse signed bound",
                ))?;
            let planes = p
                .stored_coefficients
                .checked_mul(p.num_digits)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(MetalCommitError::ShapeOverflow("recursive balanced planes"))?;
            if p.bit_width == 0 {
                self.validate_buffer_length(planes)?;
            }
            self.validate_buffer_length(size_of_val(terms).max(size_of::<RecursiveSparseTerm>()))?;
        }
        if absolute_bound > i32::MAX as u64 {
            return Err(MetalCommitError::ShapeOverflow(
                "recursive sparse fold exact signed i32 capacity",
            ));
        }
        Ok(())
    }

    pub(crate) fn dispatch_recursive_sparse_fold(
        &self,
        sources: &[(
            RecursiveFoldSource,
            RecursiveFoldParams,
            Vec<RecursiveSparseTerm>,
        )],
    ) -> Result<RecursiveFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let first = sources
                .first()
                .ok_or_else(|| {
                    MetalCommitError::UnsupportedShape("empty recursive sparse batch".into())
                })?
                .1;
            let output_count = first
                .positions_per_block
                .checked_mul(first.num_digits)
                .and_then(|n| n.checked_mul(first.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("recursive sparse output"))?;
            let output_bytes = usize::try_from(output_count)
                .ok()
                .and_then(|n| n.checked_mul(4))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive sparse output bytes",
                ))?;
            let partial_bytes =
                output_bytes
                    .checked_mul(sources.len())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive sparse partial bytes",
                    ))?;
            self.validate_buffer_length(output_bytes)?;
            self.validate_buffer_length(partial_bytes)?;
            self.admit_recursive_sparse_plans(
                sources.iter().map(|(_, p, terms)| (p, terms.as_slice())),
            )?;
            for (source, p, _) in sources {
                self.validate_recursive_fold_source(source, *p)?;
            }
            // Complete admission above precedes all uploads and command creation.
            let buffer_start = Instant::now();
            let partials = self.shared_buffer(partial_bytes)?;
            let output = self.shared_buffer(output_bytes)?;
            let mut allocation_bytes =
                partial_bytes
                    .checked_add(output_bytes)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive sparse allocation",
                    ))?;
            let mut inputs = Vec::with_capacity(sources.len());
            let mut planes = Vec::with_capacity(sources.len());
            let mut term_buffers = Vec::with_capacity(sources.len());
            let zero = [RecursiveSparseTerm {
                block: 0,
                shift: 0,
                coefficient: 0,
            }];
            for (source, p, terms) in sources {
                let (input, bytes) = self.recursive_input_buffer(source, *p)?;
                inputs.push(input);
                let plane = if p.bit_width == 0 {
                    let plane_bytes = (p.stored_coefficients * p.num_digits * 2) as usize;
                    allocation_bytes = allocation_bytes.checked_add(plane_bytes).ok_or(
                        MetalCommitError::ShapeOverflow("recursive sparse planes allocation"),
                    )?;
                    Some(self.shared_buffer(plane_bytes)?)
                } else {
                    None
                };
                planes.push(plane);
                let storage = if terms.is_empty() {
                    &zero[..]
                } else {
                    terms.as_slice()
                };
                term_buffers.push(self.shared_buffer_from_slice(storage)?);
                allocation_bytes = allocation_bytes
                    .checked_add(bytes)
                    .and_then(|n| n.checked_add(size_of_val(storage)))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive sparse inputs allocation",
                    ))?;
            }
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita recursive sparse decomposition batch");
            for (index, (_, p, terms)) in sources.iter().enumerate() {
                if let Some(plane) = &planes[index] {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.recursive_balanced_planes_pipeline);
                    encoder.set_buffer(0, Some(&inputs[index]), 0);
                    encoder.set_buffer(1, Some(plane), 0);
                    set_inline_bytes(encoder, 2, p);
                    encoder.dispatch_threads(
                        MTLSize::new(p.stored_coefficients, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                    encoder.end_encoding();
                }
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.recursive_sparse_fold_pipeline);
                encoder.set_buffer(0, Some(planes[index].as_ref().unwrap_or(&inputs[index])), 0);
                encoder.set_buffer(1, Some(&term_buffers[index]), 0);
                encoder.set_buffer(2, Some(&partials), (index * output_bytes) as u64);
                set_inline_bytes(encoder, 3, p);
                set_inline_bytes(encoder, 4, &(terms.len() as u64));
                encoder.dispatch_threads(MTLSize::new(output_count, 1, 1), MTLSize::new(128, 1, 1));
                encoder.end_encoding();
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.recursive_sparse_batch_reduce_pipeline);
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &(sources.len() as u64));
            set_inline_bytes(encoder, 3, &output_count);
            encoder.dispatch_threads(MTLSize::new(output_count, 1, 1), MTLSize::new(128, 1, 1));
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: completed fold and reduction kernels write every output i32.
            let centered_coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<i32>(), output_count as usize)
                    .to_vec()
            };
            Ok(RecursiveFoldDispatchOutcome {
                coefficients: Vec::new(),
                centered_coefficients,
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
