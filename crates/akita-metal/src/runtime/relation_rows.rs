use super::*;

#[derive(Clone, Copy)]
pub(crate) enum RelationRowsMode {
    High,
    Cyclic,
    Paired,
}

#[derive(Clone, Copy)]
enum RelationRowsSource<'a, const D: usize> {
    I8(&'a [[i8; D]]),
    I32(&'a [[i32; D]]),
}

/// Immutable borrowed input with its actual bound certified once.
pub(crate) struct RelationRowsInput<'a, const D: usize> {
    source: RelationRowsSource<'a, D>,
    bound: u64,
}

impl<'a, const D: usize> RelationRowsInput<'a, D> {
    pub(crate) fn from_i8(rows: &'a [[i8; D]], log_basis: u32) -> Result<Self, MetalCommitError> {
        if !(1..=akita_types::MAX_I8_LOG_BASIS).contains(&log_basis) {
            return Err(MetalCommitError::UnsupportedShape(
                "invalid relation digit basis".into(),
            ));
        }
        let limit = 1i16 << (log_basis - 1);
        let mut bound = 0u64;
        for &value in rows.iter().flatten() {
            if !(-limit..limit).contains(&i16::from(value)) {
                return Err(MetalCommitError::UnsupportedShape(
                    "relation digits exceed their declared basis".into(),
                ));
            }
            bound = bound.max(u64::from(value.unsigned_abs()));
        }
        Ok(Self {
            source: RelationRowsSource::I8(rows),
            bound,
        })
    }

    pub(crate) fn from_i32(rows: &'a [[i32; D]], claimed: u32) -> Self {
        let actual = rows
            .iter()
            .flatten()
            .map(|value| u64::from(value.unsigned_abs()))
            .max()
            .unwrap_or(0);
        Self {
            source: RelationRowsSource::I32(rows),
            bound: actual.max(u64::from(claimed)),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self.source {
            RelationRowsSource::I8(rows) => rows.len(),
            RelationRowsSource::I32(rows) => rows.len(),
        }
    }
    fn kind(&self) -> u64 {
        match self.source {
            RelationRowsSource::I8(_) => 1,
            RelationRowsSource::I32(_) => 4,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RelationRowsParams {
    pub(crate) ring_d: u64,
    pub(crate) ntt_size: u64,
    pub(crate) num_rows: u64,
    pub(crate) num_cols: u64,
    pub(crate) matrix_rings: u64,
    pub(crate) num_primes: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) mode: u64,
    pub(crate) rhs_kind: u64,
    pub(crate) rhs_abs_bound: u64,
}

const _: [(); 80] = [(); size_of::<RelationRowsParams>()];

impl RelationRowsParams {
    pub(crate) fn new<const D: usize>(
        rows: usize,
        matrix_rings: usize,
        mode: RelationRowsMode,
        input: &RelationRowsInput<'_, D>,
    ) -> Result<Self, MetalCommitError> {
        let ntt_size = D
            .checked_mul(2)
            .ok_or(MetalCommitError::ShapeOverflow("relation NTT size"))?;
        let planes = usize::from(matches!(mode, RelationRowsMode::Paired)) + 1;
        let output = rows
            .checked_mul(D)
            .and_then(|count| count.checked_mul(planes))
            .ok_or(MetalCommitError::ShapeOverflow("relation output"))?;
        Ok(Self {
            ring_d: D as u64,
            ntt_size: ntt_size as u64,
            num_rows: rows as u64,
            num_cols: input.len() as u64,
            matrix_rings: matrix_rings as u64,
            num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
            output_coefficients: output as u64,
            mode: match mode {
                RelationRowsMode::High => 0,
                RelationRowsMode::Cyclic => 1,
                RelationRowsMode::Paired => 2,
            },
            rhs_kind: input.kind(),
            rhs_abs_bound: input.bound,
        })
    }
}

impl MetalRuntime {
    fn relation_rows_resources(&self, ring_d: usize) -> Option<&D512LinearRelationResources> {
        match ring_d {
            64 => Some(&self.fp128_relation_rows_small_resources[0]),
            128 => Some(&self.fp128_relation_rows_small_resources[1]),
            256 => Some(&self.fp128_relation_rows_small_resources[2]),
            512 => Some(&self.fp128_d512_linear_relation_resources),
            _ => None,
        }
    }

    pub(crate) fn supports_fp128_relation_rows<const D: usize>(
        &self,
        params: RelationRowsParams,
    ) -> bool {
        if !matches!(D, 64 | 128 | 256 | 512)
            || params.ring_d != D as u64
            || params.ntt_size != 2 * D as u64
            || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
            || params.num_rows == 0
            || params.num_cols == 0
            || params.mode > 2
            || !matches!(params.rhs_kind, 1 | 4)
            || params
                .num_rows
                .checked_mul(params.num_cols)
                .is_none_or(|extent| extent > params.matrix_rings)
            || params
                .num_rows
                .checked_mul(params.ring_d)
                .and_then(|count| count.checked_mul(if params.mode == 2 { 2 } else { 1 }))
                != Some(params.output_coefficients)
            || self.relation_rows_resources(D).is_none()
        {
            return false;
        }
        let Ok(cols) = usize::try_from(params.num_cols) else {
            return false;
        };
        let capacity = CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|p| p as u128),
        );
        if !capacity.supports_modulus(
            cols,
            D,
            (-F::one()).to_canonical_u128() + 1,
            params.rhs_abs_bound,
        ) {
            return false;
        }
        let maximum = self.device.max_buffer_length();
        let matrix_bytes = params
            .matrix_rings
            .checked_mul(params.ring_d)
            .and_then(|n| n.checked_mul(size_of::<Fp128Limbs>() as u64));
        let matrix_ntt_bytes = params
            .matrix_rings
            .checked_mul(params.ntt_size)
            .and_then(|n| n.checked_mul(params.num_primes))
            .and_then(|n| n.checked_mul(size_of::<i32>() as u64));
        let rhs_bytes = params
            .num_cols
            .checked_mul(params.ring_d)
            .and_then(|n| n.checked_mul(params.rhs_kind));
        let rhs_ntt_bytes = params
            .num_cols
            .checked_mul(params.ntt_size)
            .and_then(|n| n.checked_mul(params.num_primes))
            .and_then(|n| n.checked_mul(size_of::<i32>() as u64));
        let residue_bytes = params
            .num_rows
            .checked_mul(params.ntt_size)
            .and_then(|n| n.checked_mul(params.num_primes))
            .and_then(|n| n.checked_mul(size_of::<u32>() as u64));
        let output_bytes = params
            .output_coefficients
            .checked_mul(size_of::<Fp128Limbs>() as u64);
        [
            matrix_bytes,
            matrix_ntt_bytes,
            rhs_bytes,
            rhs_ntt_bytes,
            residue_bytes,
            output_bytes,
        ]
        .into_iter()
        .all(|bytes| bytes.is_some_and(|bytes| bytes > 0 && bytes <= maximum))
            && [params.matrix_rings, params.num_cols, params.num_rows]
                .into_iter()
                .all(|n| {
                    n.checked_mul(params.num_primes)
                        .is_some_and(|groups| groups <= u64::from(u32::MAX))
                })
            && params
                .num_rows
                .checked_mul(params.ring_d)
                .is_some_and(|n| n.div_ceil(256) <= u64::from(u32::MAX))
            && self.device.max_threadgroup_memory_length() >= 1024 * size_of::<i32>() as u64
            && [
                &self.fp128_relation_rows_matrix_ntt_pipeline,
                &self.fp128_relation_rows_rhs_ntt_pipeline,
                &self.fp128_relation_rows_matvec_inverse_pipeline,
            ]
            .into_iter()
            .all(|pipeline| pipeline.max_total_threads_per_threadgroup() >= 512)
            && self
                .fp128_relation_rows_reconstruct_pipeline
                .max_total_threads_per_threadgroup()
                >= 256
    }

    pub(crate) fn prepare_fp128_relation_rows_matrix<const D: usize>(
        &self,
        matrix: &Buffer,
        matrix_rings: usize,
    ) -> Result<RecursiveCommitMatrixNttOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let params = RelationRowsParams {
                ring_d: D as u64,
                ntt_size: (2 * D) as u64,
                num_rows: 1,
                num_cols: matrix_rings as u64,
                matrix_rings: matrix_rings as u64,
                num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
                output_coefficients: D as u64,
                mode: 0,
                rhs_kind: 1,
                rhs_abs_bound: 0,
            };
            let raw_bytes = matrix_rings
                .checked_mul(D)
                .and_then(|n| n.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow("relation matrix bytes"))?;
            if !self.supports_fp128_relation_rows::<D>(params) || matrix.length() < raw_bytes as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "relation matrix transform shape is unsupported".into(),
                ));
            }
            let resources = self.relation_rows_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape("relation resources unavailable".into())
            })?;
            let start = Instant::now();
            let bytes = matrix_rings
                .checked_mul(2 * D)
                .and_then(|n| n.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .and_then(|n| n.checked_mul(size_of::<i32>()))
                .ok_or(MetalCommitError::ShapeOverflow("relation matrix spectra"))?;
            let output = self.private_buffer(bytes)?;
            let buffer_setup = start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita relation matrix linear NTT");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_relation_rows_matrix_ntt_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.limb_weights), 0);
            encoder.set_buffer(4, Some(&resources.field_moduli), 0);
            encoder.set_buffer(5, Some(&resources.fwd_twiddles), 0);
            set_inline_bytes(encoder, 6, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.matrix_rings * params.num_primes, 1, 1),
                MTLSize::new(512, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            Ok(RecursiveCommitMatrixNttOutcome {
                buffer: output,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes: bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_relation_rows<const D: usize>(
        &self,
        matrix_ntt: &Buffer,
        input: &RelationRowsInput<'_, D>,
        params: RelationRowsParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let matrix_bytes = params
                .matrix_rings
                .checked_mul(params.ntt_size)
                .and_then(|n| n.checked_mul(params.num_primes))
                .and_then(|n| n.checked_mul(size_of::<i32>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("relation matrix buffer"))?;
            if !self.supports_fp128_relation_rows::<D>(params)
                || input.len() as u64 != params.num_cols
                || input.kind() != params.rhs_kind
                || input.bound > params.rhs_abs_bound
                || matrix_ntt.length() < matrix_bytes
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "relation dispatch shape or input certificate disagrees".into(),
                ));
            }
            let resources = self.relation_rows_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape("relation resources unavailable".into())
            })?;
            let start = Instant::now();
            let (rhs, input_bytes) = match input.source {
                RelationRowsSource::I8(rows) => {
                    let buffer = self.shared_slice_buffer(rows)?;
                    let bytes = if buffer.zero_copy {
                        0
                    } else {
                        size_of_val(rows)
                    };
                    (buffer.buffer, bytes)
                }
                RelationRowsSource::I32(rows) => {
                    let buffer = self.shared_slice_buffer(rows)?;
                    let bytes = if buffer.zero_copy {
                        0
                    } else {
                        size_of_val(rows)
                    };
                    (buffer.buffer, bytes)
                }
            };
            let to_bytes = |count: u64, width: usize| {
                usize::try_from(count)
                    .ok()
                    .and_then(|n| n.checked_mul(width))
                    .ok_or(MetalCommitError::ShapeOverflow("relation allocation bytes"))
            };
            let rhs_ntt_bytes = to_bytes(
                params.num_cols * params.ntt_size * params.num_primes,
                size_of::<i32>(),
            )?;
            let residue_bytes = to_bytes(
                params.num_rows * params.ntt_size * params.num_primes,
                size_of::<u32>(),
            )?;
            let output_bytes = to_bytes(params.output_coefficients, size_of::<Fp128Limbs>())?;
            let rhs_ntt = self.private_buffer(rhs_ntt_bytes)?;
            let residues = self.private_buffer(residue_bytes)?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita exact general relation rows");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_relation_rows_rhs_ntt_pipeline);
            encoder.set_buffer(0, Some(&rhs), 0);
            encoder.set_buffer(1, Some(&rhs), 0);
            encoder.set_buffer(2, Some(&rhs_ntt), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.fwd_twiddles), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_cols * params.num_primes, 1, 1),
                MTLSize::new(512, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_relation_rows_matvec_inverse_pipeline);
            encoder.set_buffer(0, Some(matrix_ntt), 0);
            encoder.set_buffer(1, Some(&rhs_ntt), 0);
            encoder.set_buffer(2, Some(&residues), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.inv_twiddles), 0);
            encoder.set_buffer(5, Some(&resources.d_inv), 0);
            set_inline_bytes(encoder, 6, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_rows * params.num_primes, 1, 1),
                MTLSize::new(512, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_relation_rows_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new((params.num_rows * params.ring_d).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback = Instant::now();
            let count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("relation output count"))?;
            // SAFETY: the completed reconstruction initialized every output limb.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), count).to_vec()
            };
            let allocation_bytes = akita_error::checked::sum([
                input_bytes,
                rhs_ntt_bytes,
                residue_bytes,
                output_bytes,
            ])
            .ok_or(MetalCommitError::ShapeOverflow("relation total allocation"))?;
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback.elapsed(),
                },
                allocation_bytes,
            })
        })
    }
}
