use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_error::{checked, AkitaError};
use akita_prover::CpuPreparedSetup;
use akita_types::AkitaExpandedSetup;
use metal::Buffer;

use crate::field::{MetalField, F};
use crate::runtime::{MetalRuntime, RecursiveCommitParams};
use crate::MetalCommitError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MatrixCacheKey {
    ring_d: usize,
    n_a: usize,
    active_a_cols: usize,
}

struct PreparedMatrix {
    buffer: Arc<Buffer>,
    bytes: usize,
}

pub(crate) struct MatrixPreparation {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) prepare_time: Duration,
}

/// Setup-bound CPU state and lazily packed Metal matrix prefixes.
pub struct MetalPreparedSetup {
    pub(crate) cpu: CpuPreparedSetup<F>,
    pub(crate) expanded: Arc<AkitaExpandedSetup<F>>,
    matrices: Mutex<HashMap<MatrixCacheKey, PreparedMatrix>>,
    recursive_matrices: Mutex<HashMap<MatrixCacheKey, PreparedMatrix>>,
}

impl MetalPreparedSetup {
    pub(crate) fn new(cpu: CpuPreparedSetup<F>, expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            cpu,
            expanded,
            matrices: Mutex::new(HashMap::new()),
            recursive_matrices: Mutex::new(HashMap::new()),
        }
    }

    /// CPU setup state used by an adapter when a source shape is deliberately
    /// outside its Metal admission boundary.
    pub fn cpu_prepared(&self) -> &CpuPreparedSetup<F> {
        &self.cpu
    }

    pub(crate) fn matrix(
        &self,
        runtime: &MetalRuntime,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<MatrixPreparation, AkitaError> {
        self.expanded
            .shared_matrix
            .ring_view_dyn(n_a, active_a_cols, ring_d)?;
        let field_count = n_a
            .checked_mul(active_a_cols)
            .and_then(|count| count.checked_mul(ring_d))
            .ok_or_else(|| MetalCommitError::ShapeOverflow("A matrix field count").into_akita())?;
        let required_bytes = field_count
            .checked_mul(size_of::<<F as MetalField>::DeviceElement>())
            .ok_or_else(|| MetalCommitError::ShapeOverflow("A matrix bytes").into_akita())?;
        let key = MatrixCacheKey {
            ring_d,
            n_a,
            active_a_cols,
        };
        let mut matrices = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        if let Some(prepared) = matrices.get(&key) {
            return Ok(MatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }
        if let Some(prepared) = matrices
            .values()
            .filter(|prepared| prepared.bytes >= required_bytes)
            .min_by_key(|prepared| prepared.bytes)
        {
            return Ok(MatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }

        let start = Instant::now();
        let fields = self
            .expanded
            .shared_matrix
            .as_field_slice()
            .get(..field_count)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "validated Metal A-matrix prefix is unexpectedly unavailable".into(),
                )
            })?;
        let packed = fields
            .iter()
            .copied()
            .map(F::into_device)
            .collect::<Vec<_>>();
        let bytes = size_of_val(packed.as_slice());
        let buffer = Arc::new(
            runtime
                .private_buffer_from_slice(&packed)
                .map_err(MetalCommitError::into_akita)?,
        );
        let prepare_time = start.elapsed();
        matrices.insert(
            key,
            PreparedMatrix {
                buffer: buffer.clone(),
                bytes,
            },
        );
        Ok(MatrixPreparation {
            buffer,
            bytes,
            cache_hit: false,
            prepare_time,
        })
    }

    pub(crate) fn recursive_matrix<const D: usize>(
        &self,
        runtime: &MetalRuntime,
        params: RecursiveCommitParams,
    ) -> Result<MatrixPreparation, AkitaError> {
        let n_a = usize::try_from(params.num_rows)
            .map_err(|_| MetalCommitError::ShapeOverflow("recursive matrix rows").into_akita())?;
        let active_a_cols = usize::try_from(params.num_cols).map_err(|_| {
            MetalCommitError::ShapeOverflow("recursive matrix columns").into_akita()
        })?;
        self.expanded
            .shared_matrix
            .ring_view_dyn(n_a, active_a_cols, D)?;
        let key = MatrixCacheKey {
            ring_d: D,
            n_a,
            active_a_cols,
        };
        let mut matrices = self
            .recursive_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        // Prime-major transforms embed their exact matrix-ring stride; unlike
        // canonical prefixes, a larger transformed shape cannot serve this key.
        if let Some(matrix) = matrices.get(&key) {
            return Ok(MatrixPreparation {
                buffer: matrix.buffer.clone(),
                bytes: matrix.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }
        let start = Instant::now();
        let count = checked::product([n_a, active_a_cols, D]).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("recursive matrix fields").into_akita()
        })?;
        let fields = self
            .expanded
            .shared_matrix
            .as_field_slice()
            .get(..count)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive matrix prefix is unavailable".into())
            })?;
        // F's canonical storage matches Fp128Limbs (verified in field.rs).
        // Borrow it when aligned, otherwise retain a shared copy only for this
        // synchronous transform; no second raw matrix remains in the cache.
        let input = runtime
            .shared_slice_buffer(fields)
            .map_err(MetalCommitError::into_akita)?;
        let outcome = runtime
            .prepare_fp128_recursive_commit_matrix::<D>(&input.buffer, params)
            .map_err(MetalCommitError::into_akita)?;
        let buffer = Arc::new(outcome.buffer);
        let bytes = outcome.allocation_bytes;
        matrices.insert(
            key,
            PreparedMatrix {
                buffer: buffer.clone(),
                bytes,
            },
        );
        Ok(MatrixPreparation {
            buffer,
            bytes,
            cache_hit: false,
            prepare_time: start.elapsed(),
        })
    }

    /// Number of independently allocated canonical and transformed matrix prefixes.
    pub fn matrix_cache_entries(&self) -> Result<usize, MetalCommitError> {
        let canonical = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len();
        let recursive = self
            .recursive_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len();
        checked::sum([canonical, recursive])
            .ok_or(MetalCommitError::ShapeOverflow("matrix cache entries"))
    }

    /// Total bytes across independently allocated resident matrix prefixes.
    pub fn matrix_cache_bytes(&self) -> Result<usize, MetalCommitError> {
        let canonical = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        let recursive = self
            .recursive_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        checked::sum(
            canonical
                .values()
                .chain(recursive.values())
                .map(|matrix| matrix.bytes),
        )
        .ok_or(MetalCommitError::ShapeOverflow("matrix cache bytes"))
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::{MetalBackend, MetalExecutionPolicy};

    #[test]
    fn larger_resident_matrix_serves_a_smaller_reshaped_prefix() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: 512 * 16,
            },
        )
        .unwrap();
        let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = backend.prepare_setup(&setup).unwrap();
        let runtime = backend.runtime().unwrap();

        let root = prepared.matrix(runtime, 512, 1, 16).unwrap();
        assert!(!root.cache_hit);
        let outer = prepared.matrix(runtime, 64, 2, 32).unwrap();
        assert!(outer.cache_hit);
        assert_eq!(outer.prepare_time, Duration::ZERO);
        assert!(Arc::ptr_eq(&root.buffer, &outer.buffer));
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
        assert_eq!(prepared.matrix_cache_bytes().unwrap(), root.bytes);
    }
}
