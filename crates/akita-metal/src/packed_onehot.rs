use akita_error::AkitaError;
use akita_prover::compute::{CommitInnerPlan, DecomposeFoldPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, DecomposeFoldWitness, OneHotPoly, RootCommitSource};

use crate::backend::MetalBackend;
use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::PackedDecomposeFoldParams;
use crate::{MetalCommitError, MetalExecutionPolicy};

/// Borrowed row-major selectors for the packed Metal root-commit kernel.
///
/// Byte zero denotes an absent coefficient. Values in `1..onehot_k` select
/// that coefficient within the row's one-hot chunk. Logical chunks are
/// column-major through `column_capacity`; omitted suffix columns are zero.
#[derive(Clone, Copy, Debug)]
pub struct PackedOneHotCommitView<'a> {
    lanes: &'a [u8],
    active_zero_rows: &'a [u64],
    zero_column_mask: u64,
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    hot_entries: usize,
    zero_suffix_start: usize,
}

impl<'a> PackedOneHotCommitView<'a> {
    /// Validate a borrowed packed selector matrix without copying it.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
    ) -> Result<Self, AkitaError> {
        Self::new_with_active_zero_rows(onehot_k, column_capacity, num_columns, lanes, &[], 0)
    }

    /// Validate selectors whose row-zero coefficients are described by a
    /// compact active-row bitset and a fixed live-column mask.
    pub fn new_with_active_zero_rows(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            onehot_k,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            None,
            None,
            None,
        )
    }

    /// Validate structural K256 geometry while reusing an exact entry count
    /// accumulated when the selector bytes were produced.
    pub fn new_k256_with_precomputed_hot_entries(
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            256,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            Some(hot_entries),
            None,
            None,
        )
    }

    /// Reuse producer-side counts and a certified all-zero row suffix.
    pub fn new_k256_with_precomputed_metrics(
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
        zero_suffix_start: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            256,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            Some(hot_entries),
            Some(zero_suffix_start),
            None,
        )
    }

    /// Validate geometry and a producer-certified selector bound without
    /// rescanning selector bytes. The producer must supply the exact entry
    /// count, maximum selector byte, and start of the all-zero row suffix for
    /// these immutable lanes and active-zero metadata. Use the checked
    /// constructor when those certificates are unavailable.
    #[expect(
        clippy::too_many_arguments,
        reason = "selector geometry and producer certificates are independent inputs"
    )]
    pub fn new_with_precomputed_metrics(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
        zero_suffix_start: usize,
        maximum_selector: u8,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            onehot_k,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            Some(hot_entries),
            Some(zero_suffix_start),
            Some(maximum_selector),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the borrowed selector view validates each independent source dimension"
    )]
    fn new_inner(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        precomputed_hot_entries: Option<usize>,
        zero_suffix_start: Option<usize>,
        maximum_selector: Option<u8>,
    ) -> Result<Self, AkitaError> {
        if onehot_k == 0 || onehot_k > usize::from(u8::MAX) + 1 || !onehot_k.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot K={onehot_k} must be a power of two at most 256"
            )));
        }
        if column_capacity == 0 || !column_capacity.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot column capacity {column_capacity} must be a nonzero power of two"
            )));
        }
        if num_columns == 0 || num_columns > column_capacity {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot live column count {num_columns} must be in 1..={column_capacity}"
            )));
        }
        if !lanes.len().is_multiple_of(num_columns) {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot lane count {} is not divisible by {num_columns} live columns",
                lanes.len()
            )));
        }
        let num_rows = lanes.len() / num_columns;
        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot row count {num_rows} must be a nonzero power of two"
            )));
        }
        let total_field_elements = num_rows
            .checked_mul(column_capacity)
            .and_then(|chunks| chunks.checked_mul(onehot_k))
            .ok_or_else(|| {
                AkitaError::InvalidInput("packed one-hot logical size overflow".into())
            })?;
        if !total_field_elements.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot logical field length {total_field_elements} is not a power of two"
            )));
        }
        let zero_suffix_start = zero_suffix_start.unwrap_or(num_rows);
        if zero_suffix_start > num_rows {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot zero suffix starts at row {zero_suffix_start}, beyond {num_rows} rows"
            )));
        }
        let live_column_mask = if num_columns == u64::BITS as usize {
            u64::MAX
        } else {
            (1u64 << num_columns) - 1
        };
        if zero_column_mask & !live_column_mask != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot committed-zero mask {zero_column_mask:#x} exceeds {num_columns} live columns"
            )));
        }
        let expected_active_words = num_rows.div_ceil(u64::BITS as usize);
        if zero_column_mask == 0 {
            if !active_zero_rows.is_empty() {
                return Err(AkitaError::InvalidInput(
                    "packed one-hot active-zero rows require a nonzero column mask".into(),
                ));
            }
        } else if active_zero_rows.len() != expected_active_words {
            return Err(AkitaError::InvalidSize {
                expected: expected_active_words,
                actual: active_zero_rows.len(),
            });
        }
        let hot_entries = if let Some(hot_entries) = precomputed_hot_entries {
            if hot_entries > lanes.len()
                || (onehot_k != 256 && maximum_selector.is_none())
                || maximum_selector.is_some_and(|maximum| usize::from(maximum) >= onehot_k)
            {
                return Err(AkitaError::InvalidInput(
                    "precomputed packed metrics exceed the selector bound or lane count".into(),
                ));
            }
            hot_entries
        } else {
            let mut hot_entries = 0usize;
            for (position, &lane) in lanes.iter().enumerate() {
                if usize::from(lane) >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "packed one-hot lane {lane} at byte {position} is outside K={onehot_k}"
                    )));
                }
                let row = position / num_columns;
                let column = position % num_columns;
                let committed_zero = lane == 0
                    && zero_column_mask & (1u64 << column) != 0
                    && active_zero_rows
                        .get(row / u64::BITS as usize)
                        .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0);
                hot_entries += usize::from(lane != 0 || committed_zero);
            }
            hot_entries
        };
        Ok(Self {
            lanes,
            active_zero_rows,
            zero_column_mask,
            num_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            zero_suffix_start,
        })
    }

    pub(crate) fn lanes(self) -> &'a [u8] {
        self.lanes
    }

    pub(crate) fn active_zero_rows(self) -> &'a [u64] {
        self.active_zero_rows
    }

    pub(crate) fn zero_column_mask(self) -> u64 {
        self.zero_column_mask
    }

    pub(crate) fn commits_zero_at(self, row: usize, column: usize) -> bool {
        self.zero_column_mask & (1u64 << column) != 0
            && self
                .active_zero_rows
                .get(row / u64::BITS as usize)
                .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0)
    }

    pub(crate) fn num_rows(self) -> usize {
        self.num_rows
    }

    pub(crate) fn num_columns(self) -> usize {
        self.num_columns
    }

    pub(crate) fn column_capacity(self) -> usize {
        self.column_capacity
    }

    pub(crate) fn onehot_k(self) -> usize {
        self.onehot_k
    }

    pub(crate) fn hot_entries(self) -> usize {
        self.hot_entries
    }

    pub(crate) fn zero_suffix_start(self) -> usize {
        self.zero_suffix_start
    }
}

impl<const D: usize> RootCommitKernel<PackedOneHotCommitView<'_>, F, D> for MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalBackend::packed_onehot_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<PackedOneHotCommitView<'_>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot commitment requires exactly one physical source, got {}",
                sources.len()
            )));
        };
        self.commit_packed_onehot::<D>(prepared, *source, plan)
            .map(|witness| vec![witness])
    }
}

impl MetalBackend {
    /// Commit one resident packed source through the D512 rank-1 or the
    /// D128 rank-3 K256 Metal kernel, selected by the plan's ring geometry.
    pub fn commit_packed_onehot<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        if D == crate::packed_onehot_fp128_d128_rank3::RING_D {
            return self.commit_packed_onehot_d128_rank3::<D>(prepared, source, plan);
        }
        let shape = match crate::packed_onehot_fp128_d512::validate_shape::<D>(source, plan) {
            Ok(shape) => shape,
            Err(error) if self.policy() == MetalExecutionPolicy::RequireMetal => return Err(error),
            Err(_) => return self.commit_packed_onehot_cpu::<D>(prepared, source, plan),
        };
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        };
        if !runtime.supports_packed_fp128_d512_panels() {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 panel pipeline is unavailable".into(),
                )
                .into_akita()),
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        }
        crate::packed_onehot_fp128_d512::commit_validated::<D>(
            self, prepared, runtime, source, plan, shape,
        )
    }

    /// Build and cache the packed A-matrix prefix the D512 rank-1 or D128
    /// rank-3 K256 commit will use, so the commit itself does not pay `matrix_prepare` on the critical
    /// path. Meant to run concurrently with the caller's row assembly. Returns
    /// `Ok(true)` when the prefix was already resident, `Ok(false)` when this
    /// call built it or when no Metal runtime is available (a later commit then
    /// takes whatever path it would have taken anyway).
    pub fn prewarm_packed_onehot_matrix(
        &self,
        prepared: &MetalPreparedSetup,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<bool, AkitaError> {
        let Some(runtime) = self.runtime() else {
            return Ok(false);
        };
        let supported = match ring_d {
            512 => runtime.supports_packed_fp128_d512_panels(),
            128 => runtime.supports_packed_fp128_d128_rank3(),
            _ => false,
        };
        if !supported {
            return Ok(false);
        }
        let _span = tracing::info_span!(
            "MetalBackend::prewarm_packed_onehot_matrix",
            ring_d,
            n_a,
            active_a_cols
        )
        .entered();
        let matrix = prepared.matrix(runtime, ring_d, n_a, active_a_cols)?;
        Ok(matrix.cache_hit)
    }

    fn commit_packed_onehot_d128_rank3<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let shape = match crate::packed_onehot_fp128_d128_rank3::validate_shape::<D>(source, plan) {
            Ok(shape) => shape,
            Err(error) if self.policy() == MetalExecutionPolicy::RequireMetal => return Err(error),
            Err(_) => return self.commit_packed_onehot_cpu::<D>(prepared, source, plan),
        };
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        };
        if !runtime.supports_packed_fp128_d128_rank3() {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "fp128 D128 rank-3 pipeline is unavailable".into(),
                )
                .into_akita()),
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        }
        crate::packed_onehot_fp128_d128_rank3::commit_validated::<D>(
            self, prepared, runtime, source, plan, shape,
        )
    }

    fn commit_packed_onehot_cpu<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let indices = (0..source.column_capacity())
            .flat_map(|column| {
                (0..source.num_rows()).map(move |row| {
                    if column >= source.num_columns() {
                        None
                    } else {
                        let lane = source.lanes()[row * source.num_columns() + column];
                        (lane != 0 || source.commits_zero_at(row, column)).then_some(lane)
                    }
                })
            })
            .collect();
        let poly = OneHotPoly::<F, u8>::new(source.onehot_k(), indices)?;
        let view = <OneHotPoly<F, u8> as RootCommitSource<F, D>>::commit_view(&poly)?;
        let mut witnesses =
            self.cpu_backend()
                .commit_inner_group(&prepared.cpu, vec![view], plan)?;
        witnesses.pop().ok_or_else(|| {
            AkitaError::InvalidSetup("CPU packed fallback returned no commitment witness".into())
        })
    }

    /// Decompose and challenge-fold a packed K16 source at D64/128/256/512,
    /// or a K256 source at D512 (rank 1) or D128 (rank 3).
    pub fn decompose_fold_packed_onehot<const D: usize>(
        &self,
        source: PackedOneHotCommitView<'_>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        let total_start = std::time::Instant::now();
        let outcome = if source.onehot_k() == 16 {
            self.runtime()
                .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?
                .dispatch_packed_k16_decompose_fold::<D>(source, plan)
                .map_err(MetalCommitError::into_akita)?
        } else {
            if !matches!(D, 128 | 512)
                || source.onehot_k() != 256
                || source.column_capacity() != 32
                || plan.num_positions_per_block == 0
                || plan.num_digits == 0
                || !(source.num_rows() * source.onehot_k()).is_multiple_of(D)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed opening requires D512 or D128 over K256/capacity32 and nonempty output"
                        .into(),
                )
                .into_akita());
            }

            let segment_rings = source.num_rows() * source.onehot_k() / D;
            if !segment_rings.is_multiple_of(plan.num_positions_per_block) {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed opening segment is not aligned to the scheduled positions".into(),
                )
                .into_akita());
            }
            let blocks_per_column = segment_rings / plan.num_positions_per_block;
            let live_challenges = source
                .num_columns()
                .checked_mul(blocks_per_column)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed opening live challenges").into_akita()
                })?;
            let expected_challenges = source
                .column_capacity()
                .checked_mul(blocks_per_column)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed opening challenges").into_akita()
                })?;
            if plan.challenges.len() != expected_challenges {
                return Err(AkitaError::InvalidSize {
                    expected: expected_challenges,
                    actual: plan.challenges.len(),
                });
            }
            let challenge_weight = plan
                .challenges
                .first()
                .map_or(0, |challenge| challenge.positions.len());
            if challenge_weight == 0
                || plan.challenges[..live_challenges].iter().any(|challenge| {
                    challenge.positions.len() != challenge_weight
                        || challenge.coeffs.len() != challenge_weight
                })
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed opening requires a fixed nonzero challenge weight".into(),
                )
                .into_akita());
            }

            let challenge_terms =
                live_challenges
                    .checked_mul(challenge_weight)
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("packed opening challenge terms")
                            .into_akita()
                    })?;
            let mut positions = Vec::with_capacity(challenge_terms);
            let mut coefficients = Vec::with_capacity(challenge_terms);
            let dense_len = live_challenges.checked_mul(64).ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed opening dense challenges").into_akita()
            })?;
            let mut dense_subring64 = vec![0i8; dense_len];
            let mut has_subring64_embedding = true;
            let subring_stride = D / 64;
            for (challenge_index, challenge) in
                plan.challenges[..live_challenges].iter().enumerate()
            {
                challenge.validate::<D>()?;
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    positions.push(u16::try_from(position).map_err(|_| {
                        MetalCommitError::UnsupportedShape(
                            "packed opening challenge position does not fit u16".into(),
                        )
                        .into_akita()
                    })?);
                    coefficients.push(coefficient);
                    let dense_position = usize::try_from(position)
                        .ok()
                        .filter(|position| position.is_multiple_of(subring_stride))
                        .map(|position| position / subring_stride)
                        .filter(|&position| position < 64);
                    if let Some(dense_position) = dense_position {
                        let slot = challenge_index * 64 + dense_position;
                        if dense_subring64[slot] == 0 {
                            dense_subring64[slot] = coefficient;
                        } else {
                            has_subring64_embedding = false;
                        }
                    } else {
                        has_subring64_embedding = false;
                    }
                }
            }
            let dense_subring64 = has_subring64_embedding.then_some(dense_subring64);
            let output_coefficients =
                plan.num_positions_per_block.checked_mul(D).ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed opening output coefficients")
                        .into_akita()
                })?;
            let runtime = self
                .runtime()
                .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
            let params = PackedDecomposeFoldParams {
                num_rows: source.num_rows() as u64,
                num_columns: source.num_columns() as u64,
                lane_stride: source.num_columns() as u64,
                num_positions: plan.num_positions_per_block as u64,
                position_start: 0,
                blocks_per_column: blocks_per_column as u64,
                challenge_weight: challenge_weight as u64,
                output_coefficients: output_coefficients as u64,
                zero_column_mask: source.zero_column_mask(),
            };
            runtime
                .dispatch_packed_fp128_decompose_fold_streaming(
                    D,
                    source.lanes(),
                    source.active_zero_rows(),
                    &positions,
                    &coefficients,
                    dense_subring64.as_deref(),
                    params,
                    plan.num_positions_per_block,
                    |_, _| {},
                )
                .map_err(MetalCommitError::into_akita)?
        };
        let timings = outcome.timings;
        let allocation_bytes = outcome.allocation_bytes;
        let mut centered = outcome.centered_coefficients;
        let expected_coefficients = plan.num_positions_per_block.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed opening output coefficients").into_akita()
        })?;
        if centered.len() != expected_coefficients {
            return Err(AkitaError::InvalidSize {
                expected: expected_coefficients,
                actual: centered.len(),
            });
        }
        if plan.num_digits > 1 {
            let expanded_len = centered
                .len()
                .checked_mul(plan.num_digits)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed opening digit expansion").into_akita()
                })?;
            centered.resize(expanded_len, 0);
            let coefficients_per_position = D.checked_mul(plan.num_digits).ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed opening digit expansion").into_akita()
            })?;
            for position in (0..plan.num_positions_per_block).rev() {
                let source_start = position * D;
                let destination_start = position * coefficients_per_position;
                centered.copy_within(source_start..source_start + D, destination_start);
                centered[destination_start + D..destination_start + coefficients_per_position]
                    .fill(0);
            }
        }
        let witness = DecomposeFoldWitness::from_centered_coeffs_flat::<D>(centered)?;
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        tracing::debug!(
            elapsed_s = total_start.elapsed().as_secs_f64(),
            "completed packed Metal decompose-fold"
        );
        Ok(witness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certified_k16_metrics_preserve_checked_counts_and_zero_rows() {
        for active_bits in [0u64, 0b0101, 0b1111] {
            let lanes = [0, 0, 0, 15, 0, 0, 3, 0];
            let active_rows = [active_bits];
            let scanned = PackedOneHotCommitView::new_with_active_zero_rows(
                16,
                4,
                2,
                &lanes,
                &active_rows,
                0b01,
            )
            .unwrap();
            let certified = PackedOneHotCommitView::new_with_precomputed_metrics(
                16,
                4,
                2,
                &lanes,
                &active_rows,
                0b01,
                scanned.hot_entries(),
                4,
                15,
            )
            .unwrap();
            assert_eq!(certified.hot_entries(), scanned.hot_entries());
            assert_eq!(certified.lanes().as_ptr(), lanes.as_ptr());
            for row in 0..4 {
                for column in 0..2 {
                    assert_eq!(
                        certified.commits_zero_at(row, column),
                        scanned.commits_zero_at(row, column),
                    );
                }
            }
        }

        let lanes = [0u8; 8];
        let certified =
            PackedOneHotCommitView::new_with_precomputed_metrics(16, 4, 2, &lanes, &[], 0, 0, 0, 0)
                .unwrap();
        assert_eq!(certified.hot_entries(), 0);
        assert_eq!(certified.zero_suffix_start(), 0);
    }

    #[test]
    fn certified_metrics_reject_out_of_range_selector_bound() {
        let lanes = [0u8, 1, 15, 0];
        for maximum_selector in [16, u8::MAX] {
            assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
                16,
                2,
                2,
                &lanes,
                &[],
                0,
                2,
                2,
                maximum_selector,
            )
            .is_err());
        }
        assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
            16,
            2,
            2,
            &lanes,
            &[],
            0,
            5,
            2,
            15,
        )
        .is_err());
        assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
            16,
            2,
            2,
            &lanes,
            &[],
            0,
            2,
            3,
            15,
        )
        .is_err());
        assert!(PackedOneHotCommitView::new(16, 2, 2, &[0, 1, 16, 0]).is_err());
    }

    #[test]
    fn validates_borrowed_geometry_and_lanes() {
        let lanes = vec![0, 1, 255, 7, 0, 3, 2, 0];
        let view = PackedOneHotCommitView::new(256, 4, 2, &lanes).unwrap();
        assert_eq!(view.lanes().as_ptr(), lanes.as_ptr());
        assert_eq!(view.num_rows(), 4);
        assert_eq!(view.hot_entries(), 5);

        assert!(PackedOneHotCommitView::new(256, 4, 3, &lanes).is_err());
        assert!(PackedOneHotCommitView::new(128, 4, 2, &lanes).is_err());
    }

    #[test]
    fn active_rows_distinguish_committed_zero_from_absence() {
        let lanes = vec![0, 0, 0, 7, 0, 0, 3, 0];
        let view =
            PackedOneHotCommitView::new_with_active_zero_rows(16, 4, 2, &lanes, &[0b0101], 0b01)
                .unwrap();
        assert!(view.commits_zero_at(0, 0));
        assert!(!view.commits_zero_at(1, 0));
        assert!(view.commits_zero_at(2, 0));
        assert!(!view.commits_zero_at(0, 1));
        assert_eq!(view.hot_entries(), 4);

        let lanes = vec![0, 0, 0, 7, 0, 0, 3, 0];
        let scanned =
            PackedOneHotCommitView::new_with_active_zero_rows(256, 4, 2, &lanes, &[0b0101], 0b01)
                .unwrap();
        let precomputed = PackedOneHotCommitView::new_k256_with_precomputed_hot_entries(
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
        )
        .unwrap();
        assert_eq!(precomputed.hot_entries(), scanned.hot_entries());
        let with_suffix = PackedOneHotCommitView::new_k256_with_precomputed_metrics(
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
            3,
        )
        .unwrap();
        assert_eq!(with_suffix.zero_suffix_start(), 3);
        assert!(PackedOneHotCommitView::new_k256_with_precomputed_metrics(
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
            5,
        )
        .is_err());
    }
}
