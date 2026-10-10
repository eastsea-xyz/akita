use super::*;
use std::sync::Arc;

/// Exact raw-source extents; columns still use the existing [position][plane] order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PackedRecursiveCommitParams {
    pub(crate) stored_coefficients: u64,
    pub(crate) physical_coefficients: u64,
    pub(crate) ring_d: u64,
    pub(crate) bit_width: u64,
    pub(crate) num_digits: u64,
    pub(crate) log_basis: u64,
    pub(crate) threshold: Fp128Limbs,
}

pub(crate) enum RecursiveCommitInput<'a> {
    Raw(&'a [i8]),
    Packed {
        owner: Arc<[u8]>,
        params: PackedRecursiveCommitParams,
    },
}

impl RecursiveCommitInput<'_> {
    pub(crate) fn packed_params(&self) -> PackedRecursiveCommitParams {
        match self {
            Self::Raw(_) => PackedRecursiveCommitParams::default(),
            Self::Packed { params, .. } => *params,
        }
    }
}

impl MetalRuntime {
    /// This is pure admission: no input uploads or transformed-matrix preparation.
    pub(crate) fn validate_recursive_commit_input<const D: usize>(
        &self,
        source: &RecursiveCommitInput<'_>,
        params: RecursiveCommitParams,
    ) -> Result<(), MetalCommitError> {
        let capacity = params
            .num_blocks
            .checked_mul(params.num_cols)
            .and_then(|count| count.checked_mul(D as u64))
            .ok_or(MetalCommitError::ShapeOverflow(
                "recursive commit source capacity",
            ))?;
        let bytes = match source {
            RecursiveCommitInput::Raw(digits) => {
                if (digits.len() as u64) < capacity {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive raw input extent".into(),
                    ));
                }
                digits.len()
            }
            RecursiveCommitInput::Packed { owner, params: p } => {
                if p.ring_d != D as u64
                    || !(1..=8).contains(&p.bit_width)
                    || !(1..=8).contains(&p.log_basis)
                    || p.num_digits == 0
                    || !params.num_cols.is_multiple_of(p.num_digits.max(1))
                    || p.num_digits
                        .checked_mul(p.log_basis)
                        .is_none_or(|bits| bits > 128 + p.log_basis)
                    || p.physical_coefficients == 0
                    || !p.physical_coefficients.is_multiple_of(D as u64)
                    || p.stored_coefficients == 0
                    || p.stored_coefficients > p.physical_coefficients
                    || p.physical_coefficients > capacity / p.num_digits.max(1)
                    || params.rhs_abs_bound
                        < if p.num_digits == 1 {
                            1u64 << (p.bit_width - 1)
                        } else {
                            1u64 << (p.log_basis - 1)
                        }
                {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive packed input geometry".into(),
                    ));
                }
                if p.num_digits > 1
                    && p.threshold.to_u128()
                        != akita_algebra::ring::cyclotomic::decompose_centering_threshold(
                            p.num_digits as usize,
                            p.log_basis as u32,
                            (-F::one()).to_canonical_u128() + 1,
                        )
                {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive packed centering threshold".into(),
                    ));
                }
                let bytes = p
                    .stored_coefficients
                    .checked_mul(p.bit_width)
                    .map(|bits| bits.div_ceil(8))
                    .and_then(|bytes| usize::try_from(bytes).ok())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive packed input bytes",
                    ))?;
                if owner.len() < bytes {
                    return Err(MetalCommitError::UnsupportedShape(
                        "recursive packed input payload".into(),
                    ));
                }
                bytes
            }
        };
        self.validate_buffer_length(bytes)
    }

    pub(crate) fn recursive_commit_input_buffer(
        &self,
        source: &RecursiveCommitInput<'_>,
    ) -> Result<(crate::prepared::MatrixBuffer, usize), MetalCommitError> {
        match source {
            RecursiveCommitInput::Raw(digits) => {
                // Reference inputs stay independent owned copies; no borrowed
                // no-copy guard is lost when returning the owner wrapper.
                Ok((
                    self.shared_buffer_from_slice(*digits)?.into(),
                    size_of_val(*digits),
                ))
            }
            RecursiveCommitInput::Packed { owner, params } => self.packed_immutable_input_buffer(
                owner,
                params.bit_width,
                params.stored_coefficients,
            ),
        }
    }
}

const _: [(); 64] = [(); size_of::<PackedRecursiveCommitParams>()];
