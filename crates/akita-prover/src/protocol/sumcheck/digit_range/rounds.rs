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
    /// Upload the exact prefix once. Errors abort the proof rather than
    /// changing execution policy after the transcript has advanced.
    fn start(
        &self,
        input: ClassIndexedDigitTreeInput<'_, E>,
    ) -> Result<Box<dyn ClassIndexedDigitTreeSession<E>>, AkitaError>;
}

/// Resident arithmetic state for the remaining rounds of one tree stage.
pub trait ClassIndexedDigitTreeSession<E: Field>: Send + Sync {
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
