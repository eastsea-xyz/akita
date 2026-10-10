use super::*;
use crate::runtime::{FieldCoefficientPackingParams, FieldPackingSource};
use crate::MetalExecutionPolicy;
use akita_error::checked;
use akita_prover::backend::{
    RecursiveFoldBatchView, RecursiveFoldSource, RecursiveWitnessFlat, SuffixWitnessBatchView,
};
use akita_prover::compute::{CpuBackend, RootPolyMeta, SubringCoefficientPackingBatchKernel};
use jolt_field::ExtField;
use std::mem::size_of_val;

fn recursive_packing_weights<E: ExtField<F>>(
    point: &PreparedSubringCoefficientPackingPoint<E>,
) -> Result<Vec<Fp128Limbs>, AkitaError> {
    let geometry = point.geometry();
    if E::DEGREE != geometry.extension_degree()
        || point.position_weights().len() != point.num_positions_per_block()
        || point.packing_weights().len() != geometry.subring_embedding_stride()
    {
        return Err(AkitaError::InvalidSetup(
            "recursive packing weight geometry mismatch".into(),
        ));
    }
    let terms = checked::product([
        point.num_positions_per_block(),
        geometry.subring_embedding_stride(),
    ])
    .ok_or_else(|| MetalCommitError::ShapeOverflow("recursive packing terms").into_akita())?;
    let count = checked::product([terms, E::DEGREE])
        .ok_or_else(|| MetalCommitError::ShapeOverflow("recursive packing weights").into_akita())?;
    let mut weights = vec![Fp128Limbs::default(); count];
    for (position_index, &position) in point.position_weights().iter().enumerate() {
        for (packing_index, &packing) in point.packing_weights().iter().enumerate() {
            // Multiply in E before taking coordinates: degree-two cross terms
            // and its nonresidue remain exactly those of the field implementation.
            let fused = position * packing;
            let term = position_index * geometry.subring_embedding_stride() + packing_index;
            for coordinate in 0..E::DEGREE {
                weights[coordinate * terms + term] =
                    fused.base_coefficient(coordinate).into_device();
            }
        }
    }
    Ok(weights)
}

fn packing_witness_source<const D: usize>(
    witness: &RecursiveWitnessFlat,
    live_positions: usize,
) -> Result<FieldPackingSource<'_>, AkitaError> {
    witness.view::<F, D>()?;
    if witness.live_coeff_len().div_ceil(D) != live_positions {
        return Err(AkitaError::InvalidSetup(
            "recursive witness disagrees with packing live positions".into(),
        ));
    }
    let (bytes, bit_width, stored_coefficients) = witness.packed_digit_storage();
    Ok(FieldPackingSource::Packed {
        bytes,
        bit_width,
        stored_coefficients,
        live_coefficients: witness.live_coeff_len(),
    })
}

impl MetalBackend {
    fn recursive_packing_supported<E: ExtField<F>>(&self) -> Result<bool, AkitaError> {
        if matches!(E::DEGREE, 1 | 2) && self.runtime().is_some() {
            return Ok(true);
        }
        match self.policy() {
            MetalExecutionPolicy::PreferMetal => Ok(false),
            MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                "recursive coefficient packing requires Metal and degree one or two".into(),
            )
            .into_akita()),
        }
    }

    fn dispatch_recursive_packing<E: ExtField<F>, const D: usize>(
        &self,
        sources: &[FieldPackingSource<'_>],
        point: &PreparedSubringCoefficientPackingPoint<E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let geometry = point.geometry();
        let width =
            checked::product([point.num_live_blocks(), geometry.partial_base_field_width()])
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("recursive packing output").into_akita()
                })?;
        let params = FieldCoefficientPackingParams {
            live_coefficients: 0,
            bit_width: 0,
            num_live_positions: checked_u64(point.num_live_positions(), "packing live positions")?,
            positions_per_block: checked_u64(
                point.num_positions_per_block(),
                "packing positions per block",
            )?,
            num_blocks: checked_u64(point.num_live_blocks(), "packing blocks")?,
            ring_d: checked_u64(D, "packing dimension")?,
            stride: checked_u64(geometry.subring_embedding_stride(), "packing stride")?,
            subring_dimension: checked_u64(
                geometry.challenge_subring_dimension(),
                "packing subring",
            )?,
            extension_degree: checked_u64(geometry.extension_degree(), "packing extension degree")?,
            output_coefficients: checked_u64(width, "packing output coefficients")?,
        };
        let weights = recursive_packing_weights(point)?;
        let outcome = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?
            .dispatch_fp128_field_coefficient_packing(sources, &weights, params)
            .map_err(MetalCommitError::into_akita)?;
        let coordinates = decode_outcome(self, outcome)?;
        let mut coordinates = coordinates.into_iter();
        sources
            .iter()
            .map(|_| {
                SubringCoefficientPackingPartials::new(
                    geometry,
                    point.num_live_blocks(),
                    coordinates.by_ref().take(width).collect(),
                )
            })
            .collect()
    }
}

impl<E, const D: usize>
    SubringCoefficientPackingBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, E, D> for MetalBackend
where
    E: ExtField<F>,
    CpuBackend:
        for<'a> SubringCoefficientPackingBatchKernel<RecursiveFoldBatchView<'a, F, D>, F, E, D>,
{
    #[tracing::instrument(skip_all, name = "recursive_coefficient_packing_metal")]
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        if !self.recursive_packing_supported::<E>()? {
            let output = self.cpu_backend().coefficient_packing_partials_batch(
                prepared.map(|p| &p.cpu),
                source,
                plan,
            )?;
            self.record_opening_cpu_fallback(source.polys().len())
                .map_err(MetalCommitError::into_akita)?;
            return Ok(output);
        }
        let point = plan.point;
        if E::DEGREE != point.geometry().extension_degree() {
            return Err(AkitaError::InvalidSetup(
                "recursive packing extension degree mismatch".into(),
            ));
        }
        // Validate every source before preparing any device matrix.
        for poly in source.polys() {
            plan.validate::<D>(RootPolyMeta::<F>::num_vars(*poly))?;
            match poly {
                RecursiveFoldSource::SetupPrefix { .. } => {
                    let (fields, positions, blocks) = poly
                        .setup_prefix_packing_coefficients::<D>()?
                        .ok_or(AkitaError::InvalidProof)?;
                    if fields.len() / D != point.num_live_positions()
                        || positions != point.num_positions_per_block()
                        || blocks != point.num_live_blocks()
                    {
                        return Err(AkitaError::InvalidSetup(
                            "setup prefix disagrees with packing block geometry".into(),
                        ));
                    }
                }
                RecursiveFoldSource::Witness(witness) => {
                    packing_witness_source::<D>(witness, point.num_live_positions())?;
                }
            }
        }
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let mut inputs = Vec::with_capacity(source.polys().len());
        for poly in source.polys() {
            match poly {
                RecursiveFoldSource::SetupPrefix { expanded, .. } => {
                    let (fields, _, _) = poly
                        .setup_prefix_packing_coefficients::<D>()?
                        .ok_or(AkitaError::InvalidProof)?;
                    if fields.len() / D != point.num_live_positions() {
                        return Err(AkitaError::InvalidSetup(
                            "setup prefix disagrees with packing live positions".into(),
                        ));
                    }
                    let buffer = if let Some(prepared) =
                        prepared.filter(|p| std::ptr::eq(p.expanded.as_ref(), expanded.as_ref()))
                    {
                        let matrix = prepared.matrix(runtime, D, 1, fields.len() / D)?;
                        self.update_opening_metrics(|metrics| {
                            metrics.buffer_setup_time += matrix.prepare_time;
                            if !matrix.cache_hit {
                                metrics.allocation_bytes =
                                    metrics.allocation_bytes.saturating_add(matrix.bytes);
                            }
                        })
                        .map_err(MetalCommitError::into_akita)?;
                        matrix.buffer.as_ref().clone()
                    } else {
                        let buffer = runtime
                            .shared_slice_buffer(fields)
                            .map_err(MetalCommitError::into_akita)?;
                        self.update_opening_metrics(|metrics| {
                            if !buffer.zero_copy {
                                metrics.allocation_bytes =
                                    metrics.allocation_bytes.saturating_add(size_of_val(fields));
                            }
                        })
                        .map_err(MetalCommitError::into_akita)?;
                        buffer.buffer
                    };
                    inputs.push(FieldPackingSource::Field {
                        buffer,
                        coefficients: fields.len(),
                    });
                }
                RecursiveFoldSource::Witness(witness) => inputs.push(packing_witness_source::<D>(
                    witness,
                    point.num_live_positions(),
                )?),
            }
        }
        self.dispatch_recursive_packing::<E, D>(&inputs, point)
    }
}

impl<E, const D: usize>
    SubringCoefficientPackingBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D> for MetalBackend
where
    E: ExtField<F>,
    CpuBackend:
        for<'a> SubringCoefficientPackingBatchKernel<SuffixWitnessBatchView<'a, F, D>, F, E, D>,
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        if !self.recursive_packing_supported::<E>()? {
            let output = self.cpu_backend().coefficient_packing_partials_batch(
                prepared.map(|p| &p.cpu),
                source,
                plan,
            )?;
            self.record_opening_cpu_fallback(source.polys().len())
                .map_err(MetalCommitError::into_akita)?;
            return Ok(output);
        }
        if E::DEGREE != plan.point.geometry().extension_degree() {
            return Err(AkitaError::InvalidSetup(
                "recursive packing extension degree mismatch".into(),
            ));
        }
        let inputs = source
            .polys()
            .iter()
            .map(|witness| {
                plan.validate::<D>(RootPolyMeta::<F>::num_vars(*witness))?;
                packing_witness_source::<D>(witness, plan.point.num_live_positions())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.dispatch_recursive_packing::<E, D>(&inputs, plan.point)
    }
}
