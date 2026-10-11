use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field, Ring};

/// Borrowed sparse-product row: challenge i reads first_ring + i * ring_stride.
/// Missing rings contribute zero, matching the prior partial quotient view.
#[derive(Clone, Copy, Debug)]
pub struct SparseHighHalfBatch<'a, F: Field, const D: usize> {
    pub rings: &'a [CyclotomicRing<F, D>],
    pub first_ring: usize,
    pub ring_stride: usize,
    pub subtract: Option<&'a CyclotomicRing<F, D>>,
}

#[inline]
fn accumulate_small_signed<F: Field + Ring>(dst: &mut F, value: F, coefficient: i64) {
    match coefficient {
        1 => *dst += value,
        -1 => *dst -= value,
        2 => {
            *dst += value;
            *dst += value;
        }
        -2 => {
            *dst -= value;
            *dst -= value;
        }
        _ => *dst += value * F::from_i64(coefficient),
    }
}

/// Exact CPU oracle/default for the operation-shaped backend method.
pub(crate) fn cpu_sparse_high_half_batch<F, const D: usize>(
    challenges: &[SparseChallenge],
    batches: &[SparseHighHalfBatch<'_, F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Send + Sync,
{
    if D == 0
        || challenges.iter().any(|challenge| {
            challenge.positions.len() != challenge.coeffs.len()
                || challenge
                    .positions
                    .iter()
                    .any(|&position| position as usize >= D)
        })
    {
        return Err(AkitaError::InvalidInput(
            "invalid sparse high-half challenge geometry".into(),
        ));
    }
    batches
        .iter()
        .map(|batch| {
            let coefficients = cfg_fold_reduce!(
                0..challenges.len(),
                || vec![F::zero(); D],
                |mut accumulator: Vec<F>, index: usize| {
                    let ring = index
                        .checked_mul(batch.ring_stride)
                        .and_then(|offset| batch.first_ring.checked_add(offset))
                        .and_then(|index| batch.rings.get(index));
                    let Some(ring) = ring else {
                        return accumulator;
                    };
                    let challenge = &challenges[index];
                    for (&position, &coefficient) in
                        challenge.positions.iter().zip(challenge.coeffs.iter())
                    {
                        let shift = position as usize;
                        for source in (D - shift)..D {
                            accumulate_small_signed(
                                &mut accumulator[shift + source - D],
                                ring.coefficients()[source],
                                i64::from(coefficient),
                            );
                        }
                    }
                    accumulator
                },
                |mut left: Vec<F>, right: Vec<F>| {
                    for (destination, source) in left.iter_mut().zip(right) {
                        *destination += source;
                    }
                    left
                }
            );
            let mut ring = CyclotomicRing::from_slice(&coefficients);
            if let Some(subtract) = batch.subtract {
                ring -= *subtract;
            }
            Ok(ring)
        })
        .collect()
}
