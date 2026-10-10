use super::*;

impl MetalRuntime {
    pub(crate) fn admit_recursive_evaluate_plan(
        &self,
        p: RecursiveFoldParams,
        position_terms: &[RecursiveWeightTerm],
        fold_terms: &[RecursiveWeightTerm],
    ) -> Result<(u64, u64, usize), MetalCommitError> {
        self.validate_recursive_fold_params(p)?;
        if position_terms
            .iter()
            .any(|t| t.position >= p.positions_per_block || t.shift >= p.ring_d)
            || fold_terms
                .iter()
                .any(|t| t.position >= p.num_blocks || t.shift >= p.ring_d)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "recursive fold weight index".into(),
            ));
        }
        let folded_count = p
            .num_blocks
            .checked_mul(p.ring_d)
            .ok_or(MetalCommitError::ShapeOverflow("recursive folded output"))?;
        let output_count =
            folded_count
                .checked_add(p.ring_d)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive evaluation output",
                ))?;
        if output_count > u64::from(u32::MAX) {
            return Err(MetalCommitError::UnsupportedShape(
                "recursive evaluation grid".into(),
            ));
        }
        let output_bytes = usize::try_from(output_count)
            .ok()
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "recursive evaluation output bytes",
            ))?;
        self.validate_buffer_length(output_bytes)?;
        self.validate_buffer_length(
            size_of_val(position_terms).max(size_of::<RecursiveWeightTerm>()),
        )?;
        self.validate_buffer_length(size_of_val(fold_terms).max(size_of::<RecursiveWeightTerm>()))?;
        Ok((folded_count, output_count, output_bytes))
    }

    pub(crate) fn dispatch_recursive_evaluate_fold(
        &self,
        source: &RecursiveFoldSource,
        p: RecursiveFoldParams,
        position_terms: &[RecursiveWeightTerm],
        fold_terms: &[RecursiveWeightTerm],
    ) -> Result<RecursiveFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let (folded_count, output_count, output_bytes) =
                self.admit_recursive_evaluate_plan(p, position_terms, fold_terms)?;
            self.validate_recursive_fold_source(source, p)?;
            // Zero weight vectors still launch a kernel that writes every output.
            let zero = [RecursiveWeightTerm {
                weight: Fp128Limbs::default(),
                position: 0,
                shift: 0,
            }];
            let position_storage = if position_terms.is_empty() {
                &zero[..]
            } else {
                position_terms
            };
            let fold_storage = if fold_terms.is_empty() {
                &zero[..]
            } else {
                fold_terms
            };
            self.validate_buffer_length(size_of_val(position_storage))?;
            self.validate_buffer_length(size_of_val(fold_storage))?;
            let buffer_start = Instant::now();
            let (input, input_bytes) = self.recursive_input_buffer(source, p)?;
            let positions = self.shared_buffer_from_slice(position_storage)?;
            let folds = self.shared_buffer_from_slice(fold_storage)?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita recursive evaluate and fold");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.recursive_evaluate_fold_pipeline);
            encoder.set_buffer(0, Some(&input), 0);
            encoder.set_buffer(1, Some(&positions), 0);
            encoder.set_buffer(2, Some(&output), 0);
            set_inline_bytes(encoder, 3, &p);
            set_inline_bytes(encoder, 4, &(position_terms.len() as u64));
            encoder
                .dispatch_thread_groups(MTLSize::new(folded_count, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.recursive_fold_evaluation_pipeline);
            encoder.set_buffer(0, Some(&output), 0);
            encoder.set_buffer(1, Some(&folds), 0);
            encoder.set_buffer(
                2,
                Some(&output),
                folded_count * size_of::<Fp128Limbs>() as u64,
            );
            set_inline_bytes(encoder, 3, &p);
            set_inline_bytes(encoder, 4, &(fold_terms.len() as u64));
            encoder.dispatch_thread_groups(MTLSize::new(p.ring_d, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: both completed kernels write all limbs in disjoint output regions.
            let coefficients = unsafe {
                std::slice::from_raw_parts(
                    output.contents().cast::<Fp128Limbs>(),
                    output_count as usize,
                )
                .to_vec()
            };
            Ok(RecursiveFoldDispatchOutcome {
                coefficients,
                centered_coefficients: Vec::new(),
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes: output_bytes
                    + input_bytes
                    + size_of_val(position_storage)
                    + size_of_val(fold_storage),
            })
        })
    }
}
