use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_error::AkitaError;
use akita_prover::CpuPreparedSetup;
use akita_types::AkitaExpandedSetup;
use metal::Buffer;

use crate::field::{MetalField, F};
use crate::runtime::MetalRuntime;
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
}

impl MetalPreparedSetup {
    pub(crate) fn new(cpu: CpuPreparedSetup<F>, expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            cpu,
            expanded,
            matrices: Mutex::new(HashMap::new()),
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
        let buffer = Arc::new(
            runtime
                .shared_buffer_from_fields(fields)
                .map_err(MetalCommitError::into_akita)?,
        );
        let bytes = required_bytes;
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

    /// Number of independently allocated matrix prefixes resident on the device.
    pub fn matrix_cache_entries(&self) -> Result<usize, MetalCommitError> {
        Ok(self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len())
    }

    /// Total bytes across independently allocated resident matrix prefixes.
    pub fn matrix_cache_bytes(&self) -> Result<usize, MetalCommitError> {
        self.matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .values()
            .try_fold(0usize, |total, matrix| {
                total
                    .checked_add(matrix.bytes)
                    .ok_or(MetalCommitError::ShapeOverflow("matrix cache bytes"))
            })
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, OneHotPoly, RootCommitSource,
    };
    use akita_types::{FlatMatrix, SetupMatrixCapacity};
    use jolt_field::CanonicalEncoding;

    use super::*;
    use crate::field::Fp128Limbs;
    use crate::{MetalBackend, MetalExecutionPolicy};

    #[test]
    fn owned_shared_matrix_preserves_canonical_encoding_and_cpu_commitment() {
        const D: usize = 64;
        let plan = CommitInnerPlan {
            n_a: 2,
            num_positions_per_block: 2,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let field_count = plan.n_a * plan.num_positions_per_block * plan.num_digits_inner * D;
        let boundaries = [
            0,
            1,
            u128::from(u32::MAX),
            u128::from(u64::MAX),
            1u128 << 64,
            0xffff_ffff_ffff_ffff_ffff_ffff_0000_5808,
        ];
        let fields = (0..field_count)
            .map(|index| F::from_u128_checked(boundaries[index % boundaries.len()]).unwrap())
            .collect::<Vec<_>>();
        let mut setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: field_count,
            },
        )
        .unwrap();
        Arc::make_mut(&mut setup.expanded).shared_matrix =
            FlatMatrix::from_flat_data(fields.clone());
        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();
        let matrix = prepared.matrix(metal.runtime().unwrap(), D, 2, 2).unwrap();

        assert_eq!(matrix.buffer.storage_mode(), metal::MTLStorageMode::Shared);
        assert_eq!(matrix.bytes, field_count * size_of::<Fp128Limbs>());
        assert_eq!(matrix.buffer.length(), matrix.bytes as u64);
        for (index, &field) in fields.iter().enumerate() {
            // SAFETY: The owned shared buffer contains the fully initialized
            // canonical device encoding of `field_count` field elements.
            let encoded = unsafe {
                matrix
                    .buffer
                    .contents()
                    .cast::<Fp128Limbs>()
                    .add(index)
                    .read_unaligned()
            };
            assert_eq!(encoded, field.into_device());
        }

        let polynomial =
            OneHotPoly::<F, u16>::new(64, vec![Some(0), Some(63), None, Some(32)]).unwrap();
        let view =
            || <OneHotPoly<F, u16> as RootCommitSource<F, D>>::commit_view(&polynomial).unwrap();
        let expected = cpu
            .commit_inner_group(&cpu_prepared, vec![view()], plan)
            .unwrap();
        let actual = metal
            .commit_inner_group(&prepared, vec![view()], plan)
            .unwrap();
        assert_eq!(expected.len(), actual.len());
        for (cpu, metal) in expected.iter().zip(&actual) {
            assert_eq!(cpu.inner_rows, metal.inner_rows);
        }
        assert!(
            metal
                .last_commit_metrics()
                .unwrap()
                .unwrap()
                .matrix_cache_hit
        );
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    }

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
