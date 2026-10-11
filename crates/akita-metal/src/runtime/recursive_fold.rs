use super::*;
use std::sync::Arc;

mod evaluate;
mod sparse;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecursiveFoldParams {
    pub(crate) stored_coefficients: u64,
    pub(crate) live_rings: u64,
    pub(crate) physical_coefficients: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) ring_d: u64,
    pub(crate) bit_width: u64,
    pub(crate) num_digits: u64,
    pub(crate) log_basis: u64,
    pub(crate) threshold: Fp128Limbs,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecursiveWeightTerm {
    pub(crate) weight: Fp128Limbs,
    pub(crate) position: u64,
    pub(crate) shift: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecursiveSparseTerm {
    pub(crate) block: u64,
    pub(crate) shift: u32,
    pub(crate) coefficient: i32,
}

pub(crate) enum RecursiveFoldSource {
    Field {
        buffer: crate::prepared::MatrixBuffer,
    },
    Packed {
        owner: Arc<[u8]>,
    },
}

pub(crate) struct RecursivePackedCacheEntry {
    owner: Arc<[u8]>,
    bit_width: u64,
    stored_coefficients: u64,
    buffer: Buffer,
}

pub(crate) struct RecursiveFoldDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) centered_coefficients: Vec<i32>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

impl MetalRuntime {
    pub(crate) fn validate_recursive_fold_source(
        &self,
        source: &RecursiveFoldSource,
        p: RecursiveFoldParams,
    ) -> Result<(), MetalCommitError> {
        self.validate_recursive_fold_params(p)?;
        let required = match source {
            RecursiveFoldSource::Field { buffer } => {
                if p.bit_width != 0 {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive field storage kind".into(),
                    ));
                }
                let required = p
                    .stored_coefficients
                    .checked_mul(16)
                    .ok_or(MetalCommitError::ShapeOverflow("recursive field bytes"))?;
                if buffer.length() < required {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive field buffer".into(),
                    ));
                }
                required
            }
            RecursiveFoldSource::Packed { owner } => {
                if p.bit_width == 0 {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive packed storage kind".into(),
                    ));
                }
                let required = p
                    .stored_coefficients
                    .checked_mul(p.bit_width)
                    .ok_or(MetalCommitError::ShapeOverflow("recursive packed bytes"))?
                    .div_ceil(8);
                if u64::try_from(owner.len())
                    .ok()
                    .is_none_or(|len| len < required)
                {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive packed buffer".into(),
                    ));
                }
                required
            }
        };
        self.validate_buffer_length(
            usize::try_from(required)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive input bytes"))?,
        )
    }

    pub(crate) fn validate_recursive_fold_params(
        &self,
        p: RecursiveFoldParams,
    ) -> Result<(), MetalCommitError> {
        let capacity = p.num_blocks.checked_mul(p.positions_per_block).ok_or(
            MetalCommitError::ShapeOverflow("recursive fold block capacity"),
        )?;
        let physical_rings = p.physical_coefficients.checked_div(p.ring_d).unwrap_or(0);
        if !matches!(p.ring_d, 64 | 128 | 256 | 512)
            || p.stored_coefficients == 0
            || p.stored_coefficients > p.physical_coefficients
            || !p.physical_coefficients.is_multiple_of(p.ring_d.max(1))
            || p.live_rings == 0
            || p.live_rings > physical_rings
            || p.positions_per_block == 0
            || !p.positions_per_block.is_power_of_two()
            || p.num_blocks != p.live_rings.div_ceil(p.positions_per_block.max(1))
            || capacity > u64::from(u32::MAX)
            || p.physical_coefficients > u64::from(u32::MAX)
            || p.num_digits == 0
            || p.num_digits > 128
            || self
                .recursive_evaluate_fold_pipeline
                .max_total_threads_per_threadgroup()
                < 256
            || self
                .recursive_fold_evaluation_pipeline
                .max_total_threads_per_threadgroup()
                < 256
            || self
                .recursive_balanced_planes_pipeline
                .max_total_threads_per_threadgroup()
                < 128
            || self
                .recursive_sparse_fold_pipeline
                .max_total_threads_per_threadgroup()
                < 128
            || self
                .recursive_sparse_batch_reduce_pipeline
                .max_total_threads_per_threadgroup()
                < 128
        {
            return Err(MetalCommitError::UnsupportedShape(
                "recursive fold geometry".into(),
            ));
        }
        if (p.bit_width == 0 && !(1..=16).contains(&p.log_basis))
            || (p.bit_width != 0 && (!(1..=8).contains(&p.bit_width) || p.num_digits != 1))
        {
            return Err(MetalCommitError::UnsupportedShape(
                "recursive fold digit geometry".into(),
            ));
        }
        let input_bytes = if p.bit_width == 0 {
            p.stored_coefficients.checked_mul(16)
        } else {
            p.stored_coefficients
                .checked_mul(p.bit_width)
                .map(|bits| bits.div_ceil(8))
        }
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(MetalCommitError::ShapeOverflow(
            "recursive fold source bytes",
        ))?;
        self.validate_buffer_length(input_bytes)?;
        Ok(())
    }

    fn recursive_input_buffer(
        &self,
        source: &RecursiveFoldSource,
        p: RecursiveFoldParams,
    ) -> Result<(crate::prepared::MatrixBuffer, usize), MetalCommitError> {
        match source {
            RecursiveFoldSource::Field { buffer } => Ok((buffer.clone(), 0)),
            RecursiveFoldSource::Packed { owner } => {
                self.packed_immutable_input_buffer(owner, p.bit_width, p.stored_coefficients)
            }
        }
    }

    pub(super) fn packed_immutable_input_buffer(
        &self,
        owner: &Arc<[u8]>,
        bit_width: u64,
        stored_coefficients: u64,
    ) -> Result<(crate::prepared::MatrixBuffer, usize), MetalCommitError> {
        if !(1..=8).contains(&bit_width) || stored_coefficients == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "immutable packed input geometry".into(),
            ));
        }
        let payload_len = stored_coefficients
            .checked_mul(bit_width)
            .map(|bits| bits.div_ceil(8))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(MetalCommitError::ShapeOverflow(
                "immutable packed input payload",
            ))?;
        if owner.len() < payload_len {
            return Err(MetalCommitError::UnsupportedShape(
                "immutable packed input extent".into(),
            ));
        }
        self.validate_buffer_length(payload_len)?;
        let mut cache = self
            .recursive_packed_cache
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        if let Some(entry) = cache.iter().find(|entry| {
            Arc::ptr_eq(&entry.owner, owner)
                && entry.bit_width == bit_width
                && entry.stored_coefficients == stored_coefficients
        }) {
            return Ok((entry.buffer.clone().into(), 0));
        }
        let buffer = self.shared_buffer_from_slice(&owner[..payload_len])?;
        // Bounded immutable ownership: no stale pointer hits and no byte rescans.
        const MAX_CACHE_BYTES: usize = 64 * 1024 * 1024;
        if payload_len <= MAX_CACHE_BYTES {
            while cache.len() >= 8
                || cache.iter().map(|e| e.buffer.length()).sum::<u64>() + payload_len as u64
                    > MAX_CACHE_BYTES as u64
            {
                cache.remove(0);
            }
            cache.push(RecursivePackedCacheEntry {
                owner: owner.clone(),
                bit_width,
                stored_coefficients,
                buffer: buffer.clone(),
            });
        }
        Ok((buffer.into(), payload_len))
    }
}

const _: [(); 96] = [(); size_of::<RecursiveFoldParams>()];
const _: [(); 32] = [(); size_of::<RecursiveWeightTerm>()];
const _: [(); 16] = [(); size_of::<RecursiveSparseTerm>()];
