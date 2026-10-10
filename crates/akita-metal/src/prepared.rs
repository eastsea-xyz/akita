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

#[path = "prepared/matrix_buffer.rs"]
mod matrix_buffer;
pub(crate) use matrix_buffer::MatrixBuffer;
mod spectrum;
pub(crate) use spectrum::{MatrixSpectrumLayout, RecursiveMatrixPreparation, RecursiveMatrixView};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MatrixCacheKey {
    ring_d: usize,
    n_a: usize,
    active_a_cols: usize,
}

struct PreparedMatrix {
    buffer: Arc<MatrixBuffer>,
    bytes: usize,
}

pub(crate) struct MatrixPreparation {
    pub(crate) buffer: Arc<MatrixBuffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) prepare_time: Duration,
}

/// Setup-bound CPU state and lazily packed Metal matrix prefixes.
pub struct MetalPreparedSetup {
    pub(crate) cpu: CpuPreparedSetup<F>,
    pub(crate) expanded: Arc<AkitaExpandedSetup<F>>,
    matrices: Mutex<Option<PreparedMatrix>>,
    linear_matrices: Mutex<HashMap<MatrixCacheKey, PreparedMatrix>>,
}

impl MetalPreparedSetup {
    pub(crate) fn new(cpu: CpuPreparedSetup<F>, expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            cpu,
            expanded,
            matrices: Mutex::new(None),
            linear_matrices: Mutex::new(HashMap::new()),
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
        let mut matrices = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        if let Some(prepared) = matrices
            .as_ref()
            .filter(|prepared| prepared.bytes >= required_bytes)
        {
            return Ok(MatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }

        let start = Instant::now();
        // F's fixed canonical storage matches Fp128Limbs. Borrow aligned storage
        // with its setup owner, or keep exactly one shared copy for unaligned input.
        let buffer = Arc::new(MatrixBuffer::from_setup(
            runtime,
            &self.expanded,
            0..field_count,
        )?);
        let bytes = required_bytes;
        let prepare_time = start.elapsed();
        // A canonical flat prefix serves every smaller reshaped prefix. Replacing
        // the cache drops obsolete references; escaped handles retain their owner.
        *matrices = Some(PreparedMatrix {
            buffer: buffer.clone(),
            bytes,
        });
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
    ) -> Result<RecursiveMatrixPreparation, AkitaError> {
        let rows = usize::try_from(params.num_rows)
            .map_err(|_| MetalCommitError::ShapeOverflow("recursive matrix rows").into_akita())?;
        let cols = usize::try_from(params.num_cols).map_err(|_| {
            MetalCommitError::ShapeOverflow("recursive matrix columns").into_akita()
        })?;
        self.expanded.shared_matrix.ring_view_dyn(rows, cols, D)?;
        let rings = checked::product([rows, cols]).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("recursive matrix rings").into_akita()
        })?;
        if params.ring_d != D as u64 || params.matrix_rings != rings as u64 {
            return Err(AkitaError::InvalidInput(
                "recursive spectrum geometry".into(),
            ));
        }
        // One ordinary prefix serves both domains. Never retain a duplicate D
        // transform; the upper contiguous D frequencies carry its exact spectrum.
        let matrix = self.linear_matrix::<D>(runtime, rings)?;
        RecursiveMatrixPreparation::from_ordinary::<D>(matrix).map_err(MetalCommitError::into_akita)
    }

    /// Cache zero-padded 2D spectra of a flat matrix prefix. Role-specific row
    /// strides are supplied at dispatch; the buffer retains its physical ring stride.
    pub(crate) fn linear_matrix<const D: usize>(
        &self,
        runtime: &MetalRuntime,
        matrix_rings: usize,
    ) -> Result<MatrixPreparation, AkitaError> {
        self.expanded
            .shared_matrix
            .ring_view_dyn(1, matrix_rings, D)?;
        let required = checked::product([
            matrix_rings,
            2,
            D,
            crate::runtime::FP128_D512_LINEAR_RELATION_NUM_PRIMES,
            size_of::<i32>(),
        ])
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("linear matrix spectra bytes").into_akita()
        })?;
        let mut matrices = self
            .linear_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        if let Some(matrix) = matrices
            .iter()
            .filter(|(key, matrix)| key.ring_d == D && matrix.bytes >= required)
            .map(|(_, matrix)| matrix)
            .min_by_key(|matrix| matrix.bytes)
        {
            return Ok(MatrixPreparation {
                buffer: matrix.buffer.clone(),
                bytes: matrix.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }
        let start = Instant::now();
        let count = checked::product([matrix_rings, D])
            .ok_or_else(|| MetalCommitError::ShapeOverflow("linear matrix fields").into_akita())?;
        let fields = self
            .expanded
            .shared_matrix
            .as_field_slice()
            .get(..count)
            .ok_or_else(|| AkitaError::InvalidSetup("linear matrix prefix unavailable".into()))?;
        let input = runtime
            .shared_slice_buffer(fields)
            .map_err(MetalCommitError::into_akita)?;
        let outcome = runtime
            .prepare_fp128_relation_rows_matrix::<D>(&input.buffer, matrix_rings)
            .map_err(MetalCommitError::into_akita)?;
        let buffer = Arc::new(MatrixBuffer::from(outcome.buffer));
        let bytes = outcome.allocation_bytes;
        // A larger flat prefix subsumes every smaller spectrum at this D.
        // Existing borrowers retain Arc buffers until their synchronous work ends.
        matrices.retain(|key, _| key.ring_d != D);
        matrices.insert(
            MatrixCacheKey {
                ring_d: D,
                n_a: 1,
                active_a_cols: matrix_rings,
            },
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
            .is_some();
        let linear = self
            .linear_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len();
        checked::sum([usize::from(canonical), linear])
            .ok_or(MetalCommitError::ShapeOverflow("matrix cache entries"))
    }

    /// Total logical resident bytes, including canonical no-copy views.
    pub fn matrix_cache_bytes(&self) -> Result<usize, MetalCommitError> {
        let canonical = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        let linear = self
            .linear_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        checked::sum(
            canonical
                .iter()
                .chain(linear.values())
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

#[cfg(test)]
#[path = "prepared/canonical_tests.rs"]
mod canonical_tests;

#[cfg(test)]
#[path = "prepared/spectrum_tests.rs"]
mod spectrum_tests;
