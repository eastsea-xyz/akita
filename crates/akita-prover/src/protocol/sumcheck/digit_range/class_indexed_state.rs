use super::exact_prefix::ExactPrefixTable;
use super::exact_prefix::SplitEqualitySuffixMass;
use super::rounds::{ClassIndexedDigitTreeFactory, ClassIndexedDigitTreeSession};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};
use jolt_field::Field;
use std::sync::Arc;

/// Shared storage lifecycle for class-indexed early rounds and exact-prefix later rounds.
pub(super) enum ClassIndexedTableState<Compact, FirstChallengeFolded, Row: Copy> {
    Compact(Compact),
    FirstChallengeFolded(FirstChallengeFolded),
    Materialized(ExactPrefixTable<Row>),
}

pub(super) struct ResidentDigitTreeState<E: Field> {
    pub(super) session: Box<dyn ClassIndexedDigitTreeSession<E>>,
    pub(super) split_eq: GruenSplitEq<E>,
    pub(super) explicit_len: usize,
    pub(super) domain_len: usize,
    pub(super) lanes: usize,
    pub(super) next_round: usize,
    pub(super) final_value: Option<Vec<E>>,
}

/// The CPU stage owns compact early rounds. Promotion consumes it and drops
/// its host field table after the session has copied that table into owned storage.
pub(super) trait ClassIndexedHostStage<E: Field>:
    EqFactoredSumcheckInstanceProver<E> + Sized
{
    fn can_start_resident(&self) -> bool;
    fn into_resident(
        self,
        factory: &dyn ClassIndexedDigitTreeFactory<E>,
    ) -> Result<ResidentDigitTreeState<E>, AkitaError>;
    fn host_final_values(&self) -> Result<Vec<E>, AkitaError>;
}

enum DigitTreeState<P, E: Field> {
    Host(P),
    Resident(ResidentDigitTreeState<E>),
    Failed,
}

/// A fallible arithmetic adapter around the unchanged CPU tree stage.
pub(super) struct ClassIndexedDigitTreeProver<P, E: Field> {
    state: DigitTreeState<P, E>,
    factory: Option<Arc<dyn ClassIndexedDigitTreeFactory<E>>>,
    num_rounds: usize,
    degree_bound: usize,
    input_claim: E,
}

impl<P: ClassIndexedHostStage<E>, E: Field> ClassIndexedDigitTreeProver<P, E> {
    pub(super) fn new(stage: P, factory: Option<Arc<dyn ClassIndexedDigitTreeFactory<E>>>) -> Self {
        Self {
            num_rounds: EqFactoredSumcheckInstanceProver::num_rounds(&stage),
            degree_bound: EqFactoredSumcheckInstanceProver::degree_bound(&stage),
            input_claim: EqFactoredSumcheckInstanceProver::input_claim(&stage),
            state: DigitTreeState::Host(stage),
            factory,
        }
    }

    pub(super) fn final_values(&self) -> Result<Vec<E>, AkitaError> {
        match &self.state {
            DigitTreeState::Host(stage) => stage.host_final_values(),
            DigitTreeState::Resident(state) => {
                state.final_value.clone().ok_or(AkitaError::InvalidProof)
            }
            DigitTreeState::Failed => Err(AkitaError::InvalidProof),
        }
    }
}

impl<P: ClassIndexedHostStage<E>, E: Field> EqFactoredSumcheckInstanceProver<E>
    for ClassIndexedDigitTreeProver<P, E>
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn degree_bound(&self) -> usize {
        self.degree_bound
    }
    fn input_claim(&self) -> E {
        self.input_claim
    }
    fn current_tau(&self) -> E {
        match &self.state {
            DigitTreeState::Host(stage) => EqFactoredSumcheckInstanceProver::current_tau(stage),
            DigitTreeState::Resident(state) => state.split_eq.current_tau(),
            DigitTreeState::Failed => E::zero(),
        }
    }

    fn compute_round_eq_factored(
        &mut self,
        round: usize,
    ) -> Result<EqFactoredUniPoly<E>, AkitaError> {
        match &mut self.state {
            DigitTreeState::Host(stage) => {
                EqFactoredSumcheckInstanceProver::compute_round_eq_factored(stage, round)
            }
            DigitTreeState::Resident(state) => {
                if round != state.next_round || state.domain_len < 2 {
                    return Err(AkitaError::InvalidProof);
                }
                let (first, second) = state.split_eq.remaining_eq_tables();
                let suffix_weight = SplitEqualitySuffixMass::new(first, second)?
                    .weight_from(state.explicit_len.div_ceil(2))?;
                let coefficients =
                    state
                        .session
                        .round_coefficients(first, second, suffix_weight)?;
                Ok(EqFactoredUniPoly::from_q_coeffs(
                    coefficients[..=self.degree_bound].to_vec(),
                ))
            }
            DigitTreeState::Failed => Err(AkitaError::InvalidProof),
        }
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        match &mut self.state {
            DigitTreeState::Host(stage) => {
                EqFactoredSumcheckInstanceProver::ingest_challenge(stage, round, challenge)?;
                if self.factory.is_some() && stage.can_start_resident() {
                    let old = std::mem::replace(&mut self.state, DigitTreeState::Failed);
                    if let DigitTreeState::Host(stage) = old {
                        let factory = self.factory.as_ref().ok_or(AkitaError::InvalidProof)?;
                        self.state =
                            DigitTreeState::Resident(stage.into_resident(factory.as_ref())?);
                    } else {
                        return Err(AkitaError::InvalidProof);
                    }
                }
            }
            DigitTreeState::Resident(state) => {
                if round != state.next_round || state.domain_len < 2 {
                    return Err(AkitaError::InvalidProof);
                }
                let final_value = state.session.ingest_challenge(challenge)?;
                let next_domain_len = state.domain_len / 2;
                if next_domain_len == 1 {
                    let values = final_value.ok_or(AkitaError::InvalidProof)?;
                    if values.len() != state.lanes {
                        return Err(AkitaError::InvalidProof);
                    }
                    state.final_value = Some(values);
                } else if final_value.is_some() {
                    return Err(AkitaError::InvalidProof);
                }
                state.split_eq.bind(challenge);
                state.explicit_len = state.explicit_len.div_ceil(2);
                state.domain_len = next_domain_len;
                state.next_round += 1;
            }
            DigitTreeState::Failed => return Err(AkitaError::InvalidProof),
        }
        Ok(())
    }
}

impl<Compact, FirstChallengeFolded, Row: Copy>
    ClassIndexedTableState<Compact, FirstChallengeFolded, Row>
{
    pub(super) fn final_value(&self) -> Option<Row> {
        match self {
            Self::Materialized(table) => table.final_value(),
            Self::Compact(_) | Self::FirstChallengeFolded(_) => None,
        }
    }
}
