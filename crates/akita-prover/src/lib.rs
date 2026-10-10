//! Prover-facing API surface for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Config and schedule policy live in `akita-config`.

pub mod api;
pub mod backend;
pub mod compute;
pub mod kernels;
pub mod protocol;
pub mod types;
mod validation;

use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::RingVec;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;
use std::sync::OnceLock;

pub use api::{
    commit, commit_setup_prefix, resolve_polynomial_group_layout, AkitaProverSetup, CommitOutput,
    GroupContext, PreparedGroupProveOps, PreparedProverGroup,
};

pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultilinearPolynomial, OneHotIndex, OneHotPoly,
    RecursiveFoldBatchView, RecursiveFoldSource, RecursiveWitnessFlat, SparseRingBlockEntry,
    SuffixWitnessBatchView, SuffixWitnessView,
};
pub use compute::{
    planned_ntt_cache_metrics, prewarm_ntt_requirements, BatchDecomposeFoldOutcome,
    CommitBackendFor, CommitCluster, ComputeBackendSetup, CpuBackend, CpuPreparedSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend, LevelProveStacks, NttCacheOwnerId,
    NttExecutionRequirements, NttOperationCluster, OpeningCluster, OpeningProveBackendFor,
    OperationCtx, PlannedNttCacheOwnerMetric, PreparedCrtNttProfile, PreparedNttCacheMetric,
    ProveFlowBackendFor, ProveStackFor, ProverComputeStack, RecursiveProveBackend,
    ReleaseRootNttAfterFold, RingSwitchCluster, RingSwitchProveBackend, RingSwitchRelationRows,
    RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape, RootProveBackend,
    RootProvePoly, RootTensorSource, RoutedNttRequirement, RuntimeCommitBackendFor,
    RuntimeCommitSource, RuntimeOpeningProveBackendFor, RuntimeRecursiveWitnessProveBackend,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend, TensorBackendFor, TensorCluster,
    TieredProveStacks, UniformProverStack,
};
pub use protocol::sumcheck::{
    DigitRangeProver, DirectAdditionalPair, DirectAdditionalRound, DirectDigitRangeProofBackend,
    DirectDigitRangeProofInput, DirectLinearLayout, DirectLinearRound, DirectLinearSegment,
    DirectRelationRangePreparationInput, DirectRelationRangeProofBackend,
    DirectRelationRangeProofOutput, DirectRelationRangeProofState,
    DirectRelationTwoRoundPrefixData, DirectRelationTwoRoundPrefixState, LowBasisRangeCheckProver,
    RelationRangeImageProver,
};
pub use protocol::{
    batched_prove, build_relation_weight_events, commit_terminal_w, commit_w, prove, prove_suffix,
    ProveLevelOutput, RecursiveSuffixOutcome, RelationSetupSource, RelationWeightContribution,
    RelationWeightEvent, RelationWeightEventInputs, RelationWeightEvents,
    RelationWeightFactorization, RingSwitchOutput, SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
pub use types::{ProverOpeningData, SelectedProverOpeningData};

/// Prover-side output of the decompose + challenge-fold step.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::z_folded_rings_trusted`] and
/// [`Self::centered_coeffs_trusted`].
#[derive(Debug, Clone)]
pub struct DecomposeFoldWitness<F: Field> {
    /// Explicitly supplied or lazily materialized folded field rows.
    z_folded_rings: OnceLock<RingVec<F>>,
    /// Centered integer coefficients for each folded row, stored row-major flat.
    ///
    /// Hot paths borrow typed rows via [`Self::centered_coeffs_trusted`].
    centered_coeffs_flat: Vec<i32>,
    /// Smallest signed centered coefficient.
    centered_min: i32,
    /// Largest signed centered coefficient.
    centered_max: i32,
    /// Ring dimension (field coefficients per ring element), fixed at construction.
    ring_dim: usize,
}

impl<F: Field> PartialEq for DecomposeFoldWitness<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.ring_dim != other.ring_dim
            || self.centered_coeffs_flat != other.centered_coeffs_flat
            || self.centered_min != other.centered_min
            || self.centered_max != other.centered_max
        {
            return false;
        }
        let (explicit, centered) = match (self.z_folded_rings.get(), other.z_folded_rings.get()) {
            (Some(left), Some(right)) => return left == right,
            (Some(explicit), None) | (None, Some(explicit)) => {
                (explicit, &self.centered_coeffs_flat)
            }
            (None, None) => return true,
        };
        explicit.ring_dim() == self.ring_dim
            && explicit.coeff_len() == centered.len()
            && explicit
                .coeffs()
                .iter()
                .zip(centered)
                .all(|(field, &value)| *field == F::from_i64(i64::from(value)))
    }
}

impl<F: Field> Eq for DecomposeFoldWitness<F> {}

impl<F: Field> DecomposeFoldWitness<F> {
    /// Construct from owned coefficient rows at a kernel boundary.
    pub fn from_coefficient_parts<const D: usize>(
        z_folded_coeffs: Vec<[F; D]>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_coeffs.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: OnceLock::from(RingVec::from_coefficient_rows(z_folded_coeffs)),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    /// Construct from typed ring rows at a kernel boundary.
    pub fn from_parts<const D: usize>(
        z_folded_rings: Vec<CyclotomicRing<F, D>>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: OnceLock::from(RingVec::from_ring_elems(&z_folded_rings)),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    /// Retain owned centered coefficients, materializing field rows only on access.
    ///
    /// # Errors
    ///
    /// Rejects a zero dimension or an incomplete ring row.
    pub fn from_centered_coeffs_flat<const D: usize>(
        centered_coeffs_flat: Vec<i32>,
    ) -> Result<Self, AkitaError> {
        if D == 0 || !centered_coeffs_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidInput(
                "owned decompose fold buffer has inconsistent ring geometry".into(),
            ));
        }
        let (centered_rows, remainder) = centered_coeffs_flat.as_chunks::<D>();
        debug_assert!(remainder.is_empty());
        let (centered_min, centered_max) = centered_coefficient_bounds(centered_rows);
        Ok(Self {
            z_folded_rings: OnceLock::new(),
            centered_coeffs_flat,
            centered_min,
            centered_max,
            ring_dim: D,
        })
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of folded witness rows.
    pub fn row_count(&self) -> usize {
        self.centered_coeffs_flat
            .len()
            .checked_div(self.ring_dim)
            .unwrap_or(0)
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if D == 0 || self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "decompose fold witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.centered_coeffs_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.centered_coeffs_flat.len(),
            });
        }
        if let Some(rings) = self.z_folded_rings.get() {
            if !rings.can_decode_vec(D) {
                return Err(AkitaError::InvalidSize {
                    expected: D,
                    actual: rings.coeff_len(),
                });
            }
            if rings.count() != self.centered_coeffs_flat.len() / D {
                return Err(AkitaError::InvalidInput(
                    "decompose fold witness ring row count mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Borrow folded ring rows after [`Self::ensure_ring_dim`].
    pub fn z_folded_rings_trusted<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.z_folded_rings
            .get_or_init(|| {
                RingVec::from_coefficient_rows(
                    cfg_iter!(self.centered_coeffs_trusted::<D>())
                        .map(|row| row.map(|value| F::from_i64(i64::from(value))))
                        .collect(),
                )
            })
            .as_ring_slice::<D>()
    }

    /// Borrow the centered coefficients as row-major flat storage (D-free).
    pub fn centered_coeffs_flat(&self) -> &[i32] {
        &self.centered_coeffs_flat
    }

    /// Infinity norm derived from the centered coefficient buffer.
    pub fn centered_inf_norm(&self) -> u32 {
        self.centered_min
            .unsigned_abs()
            .max(self.centered_max.unsigned_abs())
    }

    /// Signed extrema derived from the centered coefficient buffer.
    pub fn centered_signed_extrema(&self) -> (i32, i32) {
        (self.centered_min, self.centered_max)
    }

    /// Borrow centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn centered_coeffs_trusted<const D: usize>(&self) -> &[[i32; D]] {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, rem) = self.centered_coeffs_flat.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        chunks
    }

    /// Owned copy of centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn centered_coeffs_owned<const D: usize>(&self) -> Vec<[i32; D]> {
        self.centered_coeffs_trusted::<D>().to_vec()
    }
}

fn centered_coefficient_bounds<const D: usize>(rows: &[[i32; D]]) -> (i32, i32) {
    if rows.is_empty() || D == 0 {
        return (0, 0);
    }
    cfg_fold_reduce!(
        rows,
        || (i32::MAX, i32::MIN),
        |(mut min, mut max), row| {
            for &coefficient in row {
                min = min.min(coefficient);
                max = max.max(coefficient);
            }
            (min, max)
        },
        |(left_min, left_max), (right_min, right_max)| {
            (left_min.min(right_min), left_max.max(right_max))
        }
    )
}

/// Prover-side output of the inner Ajtai commit step.
///
/// Ring dimension is stored by the single flat A-ring buffer. Public commit
/// parameters own the source-block and row boundaries.
pub struct CommitInnerWitness<F: Field> {
    /// Recombined inner `A * s_i` rows in `[block][A row][coefficient]` order.
    pub inner_rows: RingVec<F>,
}

impl<F: Field> CommitInnerWitness<F> {
    /// Construct from typed kernel output at a commit boundary.
    pub fn from_rows<const D: usize>(
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    ) -> Self {
        let coefficient_count = recomposed_inner_rows
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .checked_mul(D)
            .expect("trusted inner commitment output length must fit usize");
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for block in recomposed_inner_rows {
            for row in block {
                coefficients.extend_from_slice(row.coefficients());
            }
        }
        Self {
            inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)
                .expect("typed inner commitment rows have valid ring storage"),
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.inner_rows.ring_dim()
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim() != D {
            return Err(AkitaError::InvalidInput(format!(
                "commit inner witness ring_d={} does not match requested D={D}",
                self.ring_dim()
            )));
        }
        if !self.inner_rows.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.inner_rows.coeff_len(),
            });
        }
        Ok(())
    }

    /// Borrow one source block using the row count from public parameters.
    pub fn block_rows<const D: usize>(
        &self,
        block: usize,
        rows_per_block: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        let start = block
            .checked_mul(rows_per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("inner row offset overflow".into()))?;
        let end = start
            .checked_add(rows_per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("inner row end overflow".into()))?;
        self.inner_rows
            .as_ring_slice::<D>()?
            .get(start..end)
            .ok_or_else(|| AkitaError::InvalidInput("inner row block is out of range".into()))
    }

    /// Consume the inner commitment output as the persistent A-native rows.
    pub fn into_inner_rows(self) -> RingVec<F> {
        self.inner_rows
    }
}

#[cfg(test)]
mod decompose_fold_storage_tests {
    use super::*;
    use jolt_field::{Prime128Offset275, Prime64Offset59, Ring};

    fn check_lazy_field_rows<F: Field>() {
        const D: usize = 8;
        let mut seed = 0x4156_7c91_u32;
        let mut centered = Vec::new();
        for _ in 0..32 {
            let row = std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed % 4097) as i32 - 2048
            });
            centered.push(row);
            centered.push([0; D]);
            centered.push([0; D]);
        }
        centered[0][0] = i32::MIN;
        centered[0][1] = i32::MAX;
        let eager = DecomposeFoldWitness::from_coefficient_parts(
            centered
                .iter()
                .map(|row| row.map(|value| F::from_i64(i64::from(value))))
                .collect(),
            centered.clone(),
        );
        let flat = centered.into_flattened();
        let address = flat.as_ptr();
        let lazy = DecomposeFoldWitness::<F>::from_centered_coeffs_flat::<D>(flat).unwrap();
        assert_eq!(lazy.centered_coeffs_flat().as_ptr(), address);
        assert_eq!(lazy.row_count(), 96);
        assert_eq!(lazy.centered_signed_extrema(), (i32::MIN, i32::MAX));
        assert_eq!(lazy.centered_inf_norm(), 1_u32 << 31);
        assert!(lazy.z_folded_rings.get().is_none());
        lazy.ensure_ring_dim::<D>().unwrap();
        assert_eq!(lazy, eager);
        assert!(lazy.z_folded_rings.get().is_none());
        let unmaterialized_clone = lazy.clone();
        assert_eq!(
            lazy.z_folded_rings_trusted::<D>().unwrap(),
            eager.z_folded_rings_trusted::<D>().unwrap()
        );
        assert!(lazy.z_folded_rings.get().is_some());
        assert_eq!(lazy, unmaterialized_clone);
        assert!(unmaterialized_clone.z_folded_rings.get().is_none());
        assert_eq!(lazy.clone(), lazy);
        assert!(lazy.ensure_ring_dim::<4>().is_err());
    }

    #[test]
    fn centered_storage_lazily_matches_random_eager_fields_and_zero_planes() {
        check_lazy_field_rows::<Prime64Offset59>();
        check_lazy_field_rows::<Prime128Offset275>();
    }

    #[test]
    fn centered_storage_rejects_invalid_geometry_and_handles_empty_rows() {
        type F = Prime128Offset275;
        assert!(DecomposeFoldWitness::<F>::from_centered_coeffs_flat::<0>(vec![]).is_err());
        assert!(DecomposeFoldWitness::<F>::from_centered_coeffs_flat::<8>(vec![0; 7]).is_err());
        let empty = DecomposeFoldWitness::<F>::from_centered_coeffs_flat::<8>(vec![]).unwrap();
        assert_eq!(empty.row_count(), 0);
        assert_eq!(empty.centered_signed_extrema(), (0, 0));
        assert_eq!(empty.centered_inf_norm(), 0);
        assert!(empty.z_folded_rings_trusted::<8>().unwrap().is_empty());
    }

    #[test]
    fn explicit_field_rows_keep_their_value_when_centered_rows_differ() {
        type F = Prime128Offset275;
        let explicit =
            DecomposeFoldWitness::from_coefficient_parts(vec![[F::from_i64(9); 4]], vec![[2; 4]]);
        let centered =
            DecomposeFoldWitness::<F>::from_centered_coeffs_flat::<4>(vec![2; 4]).unwrap();
        assert_ne!(explicit, centered);
        assert!(centered.z_folded_rings.get().is_none());
        assert_eq!(
            explicit.z_folded_rings_trusted::<4>().unwrap()[0].coefficients(),
            &[F::from_i64(9); 4]
        );
    }

    #[test]
    fn cpu_fold_builder_leaves_centered_output_unmaterialized() {
        type F = Prime128Offset275;
        let witness = backend::poly_helpers::build_decompose_fold_witness::<F, 4>(
            vec![[-3, 0, 2, 4], [0; 4]],
            u128::MAX - 274,
        );
        assert!(witness.z_folded_rings.get().is_none());
        assert_eq!(witness.centered_signed_extrema(), (-3, 4));
        assert_eq!(
            witness.z_folded_rings_trusted::<4>().unwrap()[0].coefficients(),
            &[-3, 0, 2, 4].map(F::from_i64)
        );
    }
}
