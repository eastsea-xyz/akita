use super::*;
use akita_prover::backend::{RecursiveFoldBatchView, SuffixWitnessBatchView};
use akita_prover::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, RootOpeningSource,
};
use akita_prover::DecomposeFoldWitness;

impl MetalBackend {
    fn evaluate_recursive_input<const D: usize>(
        &self,
        prepared: Option<&MetalPreparedSetup>,
        input: FoldInput<'_>,
        p: RecursiveFoldParams,
        positions: &[RecursiveWeightTerm],
        folds: &[RecursiveWeightTerm],
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let source = self.prepare_fold_input(prepared, input, p)?;
        let outcome = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?
            .dispatch_recursive_evaluate_fold(&source, p, positions, folds)
            .map_err(MetalCommitError::into_akita)?;
        self.record_recursive_fold(&outcome)?;
        let decode_start = Instant::now();
        let coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(i, v)| F::from_device(v, i))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        let folded_count = p.num_blocks as usize * D;
        let folded = coefficients[..folded_count]
            .as_chunks::<D>()
            .0
            .iter()
            .map(|row| CyclotomicRing::from_coefficients(std::array::from_fn(|i| row[i])))
            .collect();
        let eval = CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            coefficients[folded_count + i]
        }));
        self.update_opening_metrics(|m| m.readback_time += decode_start.elapsed())
            .map_err(MetalCommitError::into_akita)?;
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_recursive_inputs<const D: usize>(
        &self,
        prepared: Option<&MetalPreparedSetup>,
        inputs: Vec<(FoldInput<'_>, RecursiveFoldParams, Vec<RecursiveSparseTerm>)>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        // The caller admitted every source before setup-cache preparation.
        let sources = inputs
            .into_iter()
            .map(|(input, p, terms)| Ok((self.prepare_fold_input(prepared, input, p)?, p, terms)))
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let outcome = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?
            .dispatch_recursive_sparse_fold(&sources)
            .map_err(MetalCommitError::into_akita)?;
        self.record_recursive_fold(&outcome)?;
        DecomposeFoldWitness::from_centered_coeffs_flat::<D>(outcome.centered_coefficients)
    }

    fn admit_recursive_inputs(
        &self,
        inputs: &[(FoldInput<'_>, RecursiveFoldParams, Vec<RecursiveSparseTerm>)],
    ) -> Result<bool, AkitaError> {
        let result = self
            .runtime()
            .ok_or(MetalCommitError::DeviceUnavailable)
            .and_then(|runtime| {
                runtime.admit_recursive_sparse_plans(
                    inputs.iter().map(|(_, p, terms)| (p, terms.as_slice())),
                )
            });
        match result {
            Ok(()) => Ok(true),
            Err(MetalCommitError::UnsupportedShape(_))
                if self.policy() == MetalExecutionPolicy::PreferMetal =>
            {
                Ok(false)
            }
            Err(error) => Err(error.into_akita()),
        }
    }

    fn admit_recursive_evaluation(
        &self,
        p: RecursiveFoldParams,
        positions: &[RecursiveWeightTerm],
        folds: &[RecursiveWeightTerm],
    ) -> Result<bool, AkitaError> {
        let result = self
            .runtime()
            .ok_or(MetalCommitError::DeviceUnavailable)
            .and_then(|runtime| runtime.admit_recursive_evaluate_plan(p, positions, folds));
        match result {
            Ok(_) => Ok(true),
            Err(MetalCommitError::UnsupportedShape(_))
                if self.policy() == MetalExecutionPolicy::PreferMetal =>
            {
                Ok(false)
            }
            Err(error) => Err(error.into_akita()),
        }
    }
}

macro_rules! recursive_fold_kernel {
    ($view:ty, $extract:ident, $field_basis:expr) => {
        impl<const D: usize> OpeningFoldKernel<$view, F, D> for MetalBackend {
            #[tracing::instrument(skip_all, name = "recursive_evaluate_fold_metal")]
            fn evaluate_and_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view,
                plan: OpeningFoldPlan<'_, F>,
            ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
                if !self.recursive_fold_supported::<D>(None)? {
                    let output = <CpuBackend as OpeningFoldKernel<$view, F, D>>::evaluate_and_fold(
                        &self.cpu_backend(),
                        prepared.map(|p| &p.cpu),
                        source,
                        plan,
                    )?;
                    self.record_opening_cpu_fallback(1)
                        .map_err(MetalCommitError::into_akita)?;
                    return Ok(output);
                }
                let (input, p) = $extract(source, fold_positions(plan))?;
                let (_, positions, folds) = fold_weights::<D>(plan, p.num_blocks as usize)?;
                if !self.admit_recursive_evaluation(p, &positions, &folds)? {
                    let output = <CpuBackend as OpeningFoldKernel<$view, F, D>>::evaluate_and_fold(
                        &self.cpu_backend(),
                        prepared.map(|p| &p.cpu),
                        source,
                        plan,
                    )?;
                    self.record_opening_cpu_fallback(1)
                        .map_err(MetalCommitError::into_akita)?;
                    return Ok(output);
                }
                self.evaluate_recursive_input(prepared, input, p, &positions, &folds)
            }

            #[tracing::instrument(skip_all, name = "recursive_decompose_fold_metal")]
            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
                let field_basis: Option<u32> = ($field_basis)(source, plan.log_basis);
                if !self.recursive_fold_supported::<D>(field_basis)? {
                    let output = <CpuBackend as OpeningFoldKernel<$view, F, D>>::decompose_fold(
                        &self.cpu_backend(),
                        prepared.map(|p| &p.cpu),
                        source,
                        plan,
                    )?;
                    self.record_opening_cpu_fallback(1)
                        .map_err(MetalCommitError::into_akita)?;
                    return Ok(output);
                }
                let (input, mut p) = $extract(source, plan.num_positions_per_block)?;
                let terms = sparse_terms::<D>(&mut p, plan)?;
                let inputs = vec![(input, p, terms)];
                if !self.admit_recursive_inputs(&inputs)? {
                    let output = <CpuBackend as OpeningFoldKernel<$view, F, D>>::decompose_fold(
                        &self.cpu_backend(),
                        prepared.map(|p| &p.cpu),
                        source,
                        plan,
                    )?;
                    self.record_opening_cpu_fallback(1)
                        .map_err(MetalCommitError::into_akita)?;
                    return Ok(output);
                }
                self.decompose_recursive_inputs::<D>(prepared, inputs)
            }
        }
    };
}

recursive_fold_kernel!(
    SuffixWitnessView<'_, F, D>,
    suffix_input,
    |_: SuffixWitnessView<'_, F, D>, _: u32| None
);
recursive_fold_kernel!(
    RecursiveFoldView<'_, F, D>,
    recursive_input,
    |source: RecursiveFoldView<'_, F, D>, bits: u32| {
        match source {
            RecursiveFoldView::SetupPrefix { .. } => Some(bits),
            RecursiveFoldView::Witness(_) => None,
        }
    }
);

macro_rules! recursive_batch_kernel {
    ($batch:ty, $opening:expr, $has_fields:expr) => {
        impl<const D: usize> OpeningBatchKernel<$batch, F, D> for MetalBackend {
            #[tracing::instrument(skip_all, name = "recursive_decompose_fold_batch_metal")]
            fn decompose_fold_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $batch,
                plan: DecomposeFoldBatchPlan<'_>,
            ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
                let polys = source.polys();
                if polys.is_empty() {
                    return Err(AkitaError::InvalidInput(
                        "batched decompose_fold requires at least one witness".into(),
                    ));
                }
                let DecomposeFoldBatchPlan::Sparse {
                    challenges,
                    num_positions_per_block,
                    num_digits,
                    log_basis,
                } = plan;
                let field_basis = ($has_fields)(source).then_some(log_basis);
                if !self.recursive_fold_supported::<D>(field_basis)? {
                    // The CPU batch returns FallbackPerPoly and does no source arithmetic.
                    // Actual per-poly fallback calls are recorded at their execution boundary.
                    return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
                }
                if !challenges.len().is_multiple_of(polys.len()) {
                    return Err(AkitaError::InvalidInput(
                        "recursive batch challenges do not partition by claim".into(),
                    ));
                }
                let blocks_per_claim = challenges.len() / polys.len();
                if blocks_per_claim == 0 {
                    return Err(AkitaError::InvalidInput(
                        "recursive batch requires challenge blocks".into(),
                    ));
                }
                let mut inputs = Vec::with_capacity(polys.len());
                for (poly, challenges) in
                    polys.iter().zip(challenges.chunks_exact(blocks_per_claim))
                {
                    let view: RecursiveFoldView<'_, F, D> = ($opening)(*poly)?;
                    let (input, mut p) = recursive_input(view, num_positions_per_block)?;
                    let terms = sparse_terms::<D>(
                        &mut p,
                        DecomposeFoldPlan {
                            challenges,
                            num_positions_per_block,
                            num_digits,
                            log_basis,
                        },
                    )?;
                    inputs.push((input, p, terms));
                }
                if !self.admit_recursive_inputs(&inputs)? {
                    return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
                }
                self.decompose_recursive_inputs::<D>(prepared, inputs)
                    .map(BatchDecomposeFoldOutcome::Fused)
            }
        }
    };
}

fn open_recursive<const D: usize>(
    poly: &akita_prover::backend::RecursiveFoldSource<F>,
) -> Result<RecursiveFoldView<'_, F, D>, AkitaError> {
    RootOpeningSource::<F, D>::opening_view(poly)
}

fn open_suffix<const D: usize>(
    poly: &akita_prover::backend::RecursiveWitnessFlat,
) -> Result<RecursiveFoldView<'_, F, D>, AkitaError> {
    poly.view::<F, D>().map(RecursiveFoldView::Witness)
}

recursive_batch_kernel!(
    RecursiveFoldBatchView<'_, F, D>,
    open_recursive::<D>,
    |source: RecursiveFoldBatchView<'_, F, D>| source.polys().iter().any(|p| matches!(
        p,
        akita_prover::backend::RecursiveFoldSource::SetupPrefix { .. }
    ))
);
recursive_batch_kernel!(
    SuffixWitnessBatchView<'_, F, D>,
    open_suffix::<D>,
    |_: SuffixWitnessBatchView<'_, F, D>| false
);
