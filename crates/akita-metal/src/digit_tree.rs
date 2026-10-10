use std::sync::Arc;
use std::time::Instant;

use akita_error::AkitaError;
use akita_prover::protocol::sumcheck::digit_range::{
    ClassIndexedDigitTreeCompactSecondRoundInput, ClassIndexedDigitTreeCompactSourceInput,
    ClassIndexedDigitTreeCompactStageInput, ClassIndexedDigitTreeFactory,
    ClassIndexedDigitTreeInput, ClassIndexedDigitTreeRound, ClassIndexedDigitTreeSession,
};
use jolt_field::{One, Zero};

use crate::field::{Fp128Limbs, F};
use crate::runtime::digit_tree::{CompactDigitTreeSource, DigitTreePipelines, DigitTreeSession};
use crate::runtime::DispatchTimings;
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

struct MetalDigitTreeFactory {
    backend: MetalBackend,
    pipelines: Arc<DigitTreePipelines>,
    max_domain_len: usize,
    compact_source: std::sync::Mutex<Option<Arc<CompactDigitTreeSource>>>,
}

struct MetalDigitTreeSession {
    backend: MetalBackend,
    resident: DigitTreeSession,
    padding_polynomial: F,
}

impl MetalBackend {
    /// Select the class-indexed accelerator before the tree transcript advances.
    pub(crate) fn digit_tree_factory(
        &self,
        domain_len: usize,
        basis: usize,
    ) -> Result<Option<Arc<dyn ClassIndexedDigitTreeFactory<F>>>, AkitaError> {
        if !matches!(basis, 16 | 32 | 64) || !domain_len.is_power_of_two() || domain_len < 4 {
            return Ok(None);
        }
        let Some(runtime) = self.runtime() else {
            return if self.policy() == MetalExecutionPolicy::PreferMetal {
                Ok(None)
            } else {
                Err(MetalCommitError::DeviceUnavailable.into_akita())
            };
        };
        let maximum_rows = domain_len / 4;
        if !runtime.supports_digit_tree_domain(domain_len) {
            return Ok(None);
        }
        let pipelines = match runtime.digit_tree_pipelines() {
            Ok(pipelines) => pipelines,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => return Ok(None),
            Err(error) => return Err(error.into_akita()),
        };
        Ok(Some(Arc::new(MetalDigitTreeFactory {
            backend: self.clone(),
            pipelines,
            max_domain_len: maximum_rows,
            compact_source: std::sync::Mutex::new(None),
        })))
    }

    fn record_digit_tree_dispatch(&self, timings: DispatchTimings) -> Result<(), AkitaError> {
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
        })
        .map_err(MetalCommitError::into_akita)
    }
}

impl MetalDigitTreeFactory {
    fn begin_stage(
        &self,
        input: ClassIndexedDigitTreeInput<'_, F>,
    ) -> Result<MetalDigitTreeSession, AkitaError> {
        if input.domain_len > self.max_domain_len {
            return Err(AkitaError::InvalidInput(
                "digit-tree table exceeds admitted domain".into(),
            ));
        }
        let setup_start = Instant::now();
        let (arity, degree, polynomial, padding_polynomial) = match &input.round {
            ClassIndexedDigitTreeRound::Product {
                arity,
                parent_weights,
            } => {
                if !matches!(arity, 2 | 4)
                    || parent_weights.is_empty()
                    || arity.checked_mul(parent_weights.len()) != Some(input.padding.len())
                {
                    return Err(AkitaError::InvalidInput(
                        "invalid digit-tree product lanes".into(),
                    ));
                }
                let padding =
                    parent_weights
                        .iter()
                        .enumerate()
                        .fold(F::zero(), |sum, (parent, &weight)| {
                            sum + weight
                                * input.padding[parent * arity..(parent + 1) * arity]
                                    .iter()
                                    .copied()
                                    .fold(F::one(), |product, lane| product * lane)
                        });
                (*arity, *arity, parent_weights.as_slice(), padding)
            }
            ClassIndexedDigitTreeRound::RangeLeaf { coefficients } => {
                if input.padding.len() != 1 || coefficients.is_empty() || coefficients.len() > 5 {
                    return Err(AkitaError::InvalidInput(
                        "invalid digit-tree range polynomial".into(),
                    ));
                }
                let padding = coefficients
                    .iter()
                    .rev()
                    .fold(F::zero(), |value, &coefficient| {
                        value * input.padding[0] + coefficient
                    });
                (0, coefficients.len() - 1, coefficients.as_slice(), padding)
            }
        };
        let padding = input
            .padding
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let polynomial = polynomial
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let resident = runtime
            .begin_digit_tree(
                self.pipelines.clone(),
                input.explicit,
                &padding,
                &polynomial,
                input.domain_len,
                arity,
                degree,
            )
            .map_err(MetalCommitError::into_akita)?;
        self.backend
            .update_opening_metrics(|metrics| {
                metrics.buffer_setup_time += setup_start.elapsed();
                metrics.allocation_bytes = metrics
                    .allocation_bytes
                    .saturating_add(resident.allocation_bytes);
            })
            .map_err(MetalCommitError::into_akita)?;
        Ok(MetalDigitTreeSession {
            backend: self.backend.clone(),
            resident,
            padding_polynomial,
        })
    }
}

impl ClassIndexedDigitTreeFactory<F> for MetalDigitTreeFactory {
    fn start(
        &self,
        input: ClassIndexedDigitTreeInput<'_, F>,
    ) -> Result<Box<dyn ClassIndexedDigitTreeSession<F>>, AkitaError> {
        Ok(Box::new(self.begin_stage(input)?))
    }

    fn prepare_compact_source(
        &self,
        input: ClassIndexedDigitTreeCompactSourceInput<'_>,
    ) -> Result<(), AkitaError> {
        if input.domain_len == 4 {
            // This tree stays on the existing compact CPU path: a quarter domain has one row.
            *self
                .compact_source
                .lock()
                .map_err(|_| MetalCommitError::PoisonedLock.into_akita())? = None;
            return Ok(());
        }
        if input.domain_len / 4 > self.max_domain_len {
            return Err(AkitaError::InvalidProof);
        }
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let setup = Instant::now();
        let source = runtime
            .prepare_compact_digit_tree_source(input)
            .map_err(MetalCommitError::into_akita)?;
        self.backend
            .update_opening_metrics(|metrics| {
                metrics.buffer_setup_time += setup.elapsed();
                metrics.allocation_bytes = metrics
                    .allocation_bytes
                    .saturating_add(source.allocation_bytes);
            })
            .map_err(MetalCommitError::into_akita)?;
        *self
            .compact_source
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())? = Some(Arc::new(source));
        Ok(())
    }

    fn start_compact(
        &self,
        input: ClassIndexedDigitTreeCompactStageInput<'_, F>,
    ) -> Result<Option<Box<dyn ClassIndexedDigitTreeSession<F>>>, AkitaError> {
        let Some(source) = self
            .compact_source
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?
            .clone()
        else {
            return Ok(None);
        };
        let mut stage = self.begin_stage(ClassIndexedDigitTreeInput {
            domain_len: source.domain_len / 4,
            explicit: &[],
            padding: input.padding,
            round: input.round,
        })?;
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let setup = Instant::now();
        let extra = runtime
            .attach_compact_digit_tree(&mut stage.resident, source, input.pair_coefficients)
            .map_err(MetalCommitError::into_akita)?;
        self.backend
            .update_opening_metrics(|metrics| {
                metrics.buffer_setup_time += setup.elapsed();
                metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(extra);
            })
            .map_err(MetalCommitError::into_akita)?;
        Ok(Some(Box::new(stage)))
    }
}

impl ClassIndexedDigitTreeSession<F> for MetalDigitTreeSession {
    fn prepare_compact_second_round(
        &mut self,
        input: ClassIndexedDigitTreeCompactSecondRoundInput<'_, F>,
    ) -> Result<(), AkitaError> {
        let setup = Instant::now();
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let extra = runtime
            .prepare_compact_digit_tree_second_round(&mut self.resident, input)
            .map_err(MetalCommitError::into_akita)?;
        self.backend
            .update_opening_metrics(|metrics| {
                metrics.buffer_setup_time += setup.elapsed();
                metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(extra);
            })
            .map_err(MetalCommitError::into_akita)?;
        Ok(())
    }

    fn round_coefficients(
        &mut self,
        equality_first: &[F],
        equality_second: &[F],
        suffix_weight: F,
    ) -> Result<[F; 5], AkitaError> {
        let first = equality_first
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let second = equality_second
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let outcome = runtime
            .digit_tree_round(&mut self.resident, &first, &second)
            .map_err(MetalCommitError::into_akita)?;
        let mut coefficients = [F::zero(); 5];
        for (index, value) in outcome.coefficients.into_iter().enumerate() {
            coefficients[index] = value
                .into_field(index)
                .map_err(MetalCommitError::into_akita)?;
        }
        // Equal padding endpoints have zero slope: only q's constant term receives suffix mass.
        coefficients[0] += suffix_weight * self.padding_polynomial;
        self.backend.record_digit_tree_dispatch(outcome.timings)?;
        Ok(coefficients)
    }

    fn ingest_challenge(&mut self, challenge: F) -> Result<Option<Vec<F>>, AkitaError> {
        let runtime = self
            .backend
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let outcome = runtime
            .digit_tree_bind(&mut self.resident, Fp128Limbs::from_field(challenge))
            .map_err(MetalCommitError::into_akita)?;
        let final_value = outcome
            .final_value
            .map(|values| {
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            .into_field(index)
                            .map_err(MetalCommitError::into_akita)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        self.backend.record_digit_tree_dispatch(outcome.timings)?;
        Ok(final_value)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod compact_tests;
