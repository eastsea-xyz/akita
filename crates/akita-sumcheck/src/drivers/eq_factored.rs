//! Eq-factored sumcheck transcript drivers.

use crate::traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState,
};
use crate::types::{EqFactoredSumcheckProof, EqFactoredUniPoly};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, Field};

/// Advance the normalized claim for one eq-factored sumcheck round.
#[doc(hidden)]
#[inline]
pub fn advance_eq_factored_claim<E: Field>(
    claim: E,
    tau: E,
    poly: &EqFactoredUniPoly<E>,
    r_round: E,
) -> E {
    let q_0 = claim - tau * poly.nonconstant_term_sum_at_one();
    q_0 + poly.eval_nonconstant_terms(&r_round)
}

/// Plain extension for eq-factored sumcheck provers.
pub trait EqFactoredSumcheckInstanceProverExt<E>:
    EqFactoredSumcheckInstanceProver<E> + Sized
where
    E: Field,
{
    /// Produce an eq-factored sumcheck proof.
    ///
    /// The prover sends the inner polynomial `q(X)` with its constant coefficient
    /// omitted in every round. The driver recovers that coefficient by subtraction
    /// and maintains a normalized claim without division.
    ///
    /// # Errors
    ///
    /// Returns an error if any generated round polynomial exceeds the instance's
    /// degree bound or the instance fails to compute, bind, or finalize a round.
    #[tracing::instrument(skip_all, name = "prove_eq_factored_sumcheck")]
    #[inline(never)]
    fn prove<F, T, S>(
        &mut self,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
    where
        F: Field + CanonicalEncoding,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> Result<E, AkitaError>,
    {
        let num_rounds = self.num_rounds();
        let degree_bound = self.degree_bound();
        let mut claim = self.input_claim();
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        for round in 0..num_rounds {
            let poly = self.compute_round_eq_factored(round)?;
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
            let r_i = sample_challenge(transcript)?;
            claim = advance_eq_factored_claim(claim, self.current_tau(), &poly, r_i);
            challenges.push(r_i);
            self.ingest_challenge(round, r_i)?;
            round_polys.push(poly);
        }

        self.finalize()?;
        Ok((EqFactoredSumcheckProof { round_polys }, challenges, claim))
    }
}

impl<E, Inst> EqFactoredSumcheckInstanceProverExt<E> for Inst
where
    E: Field,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
}

/// Plain extension for eq-factored sumcheck verifiers.
pub trait EqFactoredSumcheckInstanceVerifierExt<E>:
    EqFactoredSumcheckInstanceVerifier<E> + Sized
where
    E: Field,
{
    /// Verify an eq-factored sumcheck proof.
    ///
    /// The verifier absorbs each round message, samples the corresponding
    /// challenge, updates the normalized running claim from the current equality
    /// coordinate and the transmitted `q(X)` data, and finally checks the
    /// expected folded oracle value at the full challenge point.
    ///
    /// This creates and owns the mutable eq-factored round state locally, while
    /// keeping `self` immutable.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof length is invalid, a round polynomial exceeds
    /// the verifier degree bound, or the final folded oracle value does not match.
    #[tracing::instrument(skip_all, name = "verify_eq_factored_sumcheck")]
    #[inline(never)]
    fn verify<F, T, S>(
        &self,
        proof: &EqFactoredSumcheckProof<E>,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> Result<E, AkitaError>,
    {
        let num_rounds = self.num_rounds();
        if proof.round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: proof.round_polys.len(),
            });
        }

        let degree_bound = self.degree_bound();
        let mut claim = self.input_claim();
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut round_state = self.start_round_state()?;

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        for (round, poly) in proof.round_polys.iter().enumerate() {
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript)?;
            claim = advance_eq_factored_claim(claim, round_state.current_tau(), poly, r_i);
            challenges.push(r_i);
            round_state.ingest_challenge(round, r_i);
        }

        let expected = self.expected_output_claim(&round_state, &challenges)?;
        if claim != expected {
            return Err(AkitaError::InvalidProof);
        }
        Ok(challenges)
    }
}

impl<E, Inst> EqFactoredSumcheckInstanceVerifierExt<E> for Inst
where
    E: Field,
    Inst: EqFactoredSumcheckInstanceVerifier<E>,
{
}

#[cfg(test)]
mod fallible_backend_tests {
    use super::*;
    use akita_transcript::AkitaTranscript;
    use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

    #[derive(Clone, Copy)]
    enum Failure {
        Round,
        Bind,
        Finalize,
    }

    struct FailingBackendProver {
        failure: Failure,
        bound: usize,
    }

    impl EqFactoredSumcheckInstanceProver<F> for FailingBackendProver {
        fn num_rounds(&self) -> usize {
            2
        }
        fn degree_bound(&self) -> usize {
            2
        }
        fn input_claim(&self) -> F {
            F::zero()
        }
        fn current_tau(&self) -> F {
            F::one()
        }
        fn compute_round_eq_factored(
            &mut self,
            _round: usize,
        ) -> Result<EqFactoredUniPoly<F>, AkitaError> {
            if matches!(self.failure, Failure::Round) {
                Err(AkitaError::InvalidInput(
                    "test accelerator round failure".into(),
                ))
            } else {
                Ok(EqFactoredUniPoly::from_q_coeffs(vec![F::zero(); 3]))
            }
        }
        fn ingest_challenge(&mut self, _round: usize, _challenge: F) -> Result<(), AkitaError> {
            if matches!(self.failure, Failure::Bind) {
                Err(AkitaError::InvalidInput(
                    "test accelerator bind failure".into(),
                ))
            } else {
                self.bound += 1;
                Ok(())
            }
        }
        fn finalize(&mut self) -> Result<(), AkitaError> {
            Err(AkitaError::InvalidInput(
                "test accelerator finalize failure".into(),
            ))
        }
    }

    #[test]
    fn accelerator_round_failure_stops_before_challenge_sampling() {
        assert_failure(Failure::Round, 0, 0);
    }

    #[test]
    fn accelerator_bind_failure_stops_before_next_round() {
        assert_failure(Failure::Bind, 1, 0);
    }

    #[test]
    fn accelerator_finalize_failure_is_propagated() {
        assert_failure(Failure::Finalize, 2, 2);
    }

    fn assert_failure(failure: Failure, expected_samples: usize, expected_binds: usize) {
        let mut prover = FailingBackendProver { failure, bound: 0 };
        let mut transcript = AkitaTranscript::<F>::new(b"eq-factored-fallible-kernel");
        let mut samples = 0;
        let result = prover.prove::<F, _, _>(&mut transcript, |_| {
            samples += 1;
            Ok(F::one())
        });
        assert!(result.is_err());
        assert_eq!(samples, expected_samples);
        assert_eq!(prover.bound, expected_binds);
    }
}
