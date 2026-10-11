//! Transcript-free, resident tails for class-indexed digit-range sumchecks.

use akita_error::AkitaError;
use jolt_field::Field;

/// Arithmetic for a materialized tree stage. The tree and its transcript stay
/// in the canonical prover; the session only evaluates this polynomial.
#[derive(Clone, Debug)]
pub enum ClassIndexedDigitTreeRound<E: Field> {
    /// A weighted sum of products of adjacent child-node lanes.
    Product {
        arity: usize,
        parent_weights: Vec<E>,
    },
    /// The plan's batched range polynomial evaluated on one range-image lane.
    RangeLeaf { coefficients: Vec<E> },
}

/// Borrowed exact-prefix state at the resident-session boundary.
pub struct ClassIndexedDigitTreeInput<'a, E: Field> {
    /// Full remaining Boolean domain, including the implicit suffix.
    pub domain_len: usize,
    /// Row-major explicit prefix; each row has `padding.len()` lanes.
    pub explicit: &'a [E],
    /// Value of every implicit row.
    pub padding: &'a [E],
    /// Arithmetic of this tree stage.
    pub round: ClassIndexedDigitTreeRound<E>,
}

/// A factory admitted before the tree starts mutating its transcript.
pub trait ClassIndexedDigitTreeFactory<E: Field>: Send + Sync {
    /// Upload compact ordered pair IDs once before this tree's transcript advances.
    fn prepare_compact_source(
        &self,
        _input: ClassIndexedDigitTreeCompactSourceInput<'_>,
    ) -> Result<(), AkitaError> {
        Ok(())
    }

    /// Start the first two rounds from small CPU class LUTs, if admitted.
    fn start_compact(
        &self,
        _input: ClassIndexedDigitTreeCompactStageInput<'_, E>,
    ) -> Result<Option<Box<dyn ClassIndexedDigitTreeSession<E>>>, AkitaError> {
        Ok(None)
    }

    /// Upload the exact prefix once. Errors abort the proof rather than
    /// changing execution policy after the transcript has advanced.
    fn start(
        &self,
        input: ClassIndexedDigitTreeInput<'_, E>,
    ) -> Result<Box<dyn ClassIndexedDigitTreeSession<E>>, AkitaError>;
}

/// Resident arithmetic state for the remaining rounds of one tree stage.
pub trait ClassIndexedDigitTreeSession<E: Field>: Send + Sync {
    /// Upload bounded first-challenge-folded class LUTs before computing round one.
    fn prepare_compact_second_round(
        &mut self,
        _input: ClassIndexedDigitTreeCompactSecondRoundInput<'_, E>,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidProof)
    }

    /// Return ordered q coefficients, including the constant coefficient.
    fn round_coefficients(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
        suffix_weight: E,
    ) -> Result<[E; 5], AkitaError>;

    /// Bind the next variable. Only the final row crosses back to the host.
    fn ingest_challenge(&mut self, challenge: E) -> Result<Option<Vec<E>>, AkitaError>;
}

/// Checked compact class source shared by every stage in one tree.
pub struct ClassIndexedDigitTreeCompactSourceInput<'a> {
    /// Full Boolean domain before the first two rounds.
    pub domain_len: usize,
    /// Number of explicit digit rows; remaining rows use class zero.
    pub live_len: usize,
    /// Number of canonical range-image collision classes.
    pub class_count: usize,
    /// Consecutive ordered class IDs, including class-zero padding in an odd pair.
    pub ordered_pairs: &'a [u16],
}

/// Small first-round LUT and arithmetic for one compact stage.
pub struct ClassIndexedDigitTreeCompactStageInput<'a, E: Field> {
    /// Ordered first-round coefficients, indexed by left * class_count + right.
    pub pair_coefficients: &'a [[E; 5]],
    /// Constant value of each implicit node lane.
    pub padding: &'a [E],
    /// Product or range-leaf arithmetic for subsequent rounds.
    pub round: ClassIndexedDigitTreeRound<E>,
}

/// First-challenge-folded lanes and optional bounded quartet coefficient LUT.
pub struct ClassIndexedDigitTreeCompactSecondRoundInput<'a, E: Field> {
    /// Row-major first-challenge-folded node lanes for every ordered class pair.
    pub folded_pairs: &'a [E],
    /// Optional <=4,096-row coefficient LUT; larger class sets use factorized kernels.
    pub quartet_coefficients: Option<&'a [[E; 5]]>,
}
