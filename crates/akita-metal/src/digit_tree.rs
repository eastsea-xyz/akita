use std::sync::Arc;
use std::time::Instant;

use akita_error::AkitaError;
use akita_prover::protocol::sumcheck::digit_range::{
    ClassIndexedDigitTreeFactory, ClassIndexedDigitTreeInput, ClassIndexedDigitTreeRound,
    ClassIndexedDigitTreeSession,
};
use jolt_field::{One, Zero};

use crate::field::{Fp128Limbs, F};
use crate::runtime::digit_tree::{DigitTreePipelines, DigitTreeSession};
use crate::runtime::DispatchTimings;
use crate::{MetalBackend, MetalCommitError, MetalExecutionPolicy};

struct MetalDigitTreeFactory {
    backend: MetalBackend,
    pipelines: Arc<DigitTreePipelines>,
    max_domain_len: usize,
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

impl ClassIndexedDigitTreeFactory<F> for MetalDigitTreeFactory {
    fn start(
        &self,
        input: ClassIndexedDigitTreeInput<'_, F>,
    ) -> Result<Box<dyn ClassIndexedDigitTreeSession<F>>, AkitaError> {
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
        Ok(Box::new(MetalDigitTreeSession {
            backend: self.backend.clone(),
            resident,
            padding_polynomial,
        }))
    }
}

impl ClassIndexedDigitTreeSession<F> for MetalDigitTreeSession {
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
