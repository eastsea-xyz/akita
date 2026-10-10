use std::sync::Arc;

use super::*;

const THREADS: usize = 256;
const MAX_WORKGROUPS: usize = 4_096;
const COEFFICIENTS: usize = 5;

pub(crate) struct DigitTreePipelines {
    partials: ComputePipelineState,
    reduce: ComputePipelineState,
    bind: ComputePipelineState,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DigitTreeParams {
    explicit_rows: u64,
    pair_count: u64,
    lanes: u64,
    first_len: u64,
    parent_count: u64,
    arity: u64,
    degree: u64,
    workgroups: u64,
}

pub(crate) struct DigitTreeSession {
    pipelines: Arc<DigitTreePipelines>,
    buffers: [Buffer; 2],
    current: usize,
    padding: Buffer,
    polynomial: Buffer,
    partials: Buffer,
    output: Buffer,
    params: DigitTreeParams,
    domain_len: usize,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DigitTreeRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
}

pub(crate) struct DigitTreeBindOutcome {
    pub(crate) final_value: Option<Vec<Fp128Limbs>>,
    pub(crate) timings: DispatchTimings,
}

impl MetalRuntime {
    pub(crate) fn supports_digit_tree_domain(&self, domain_len: usize) -> bool {
        if !domain_len.is_power_of_two() || domain_len < 4 {
            return false;
        }
        (domain_len / 4)
            .checked_mul(8)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .is_some_and(|bytes| {
                bytes as u64 <= self.device.max_buffer_length()
                    && bytes / size_of::<Fp128Limbs>() <= u32::MAX as usize
            })
    }

    pub(crate) fn digit_tree_pipelines(&self) -> Result<Arc<DigitTreePipelines>, MetalCommitError> {
        let mut cached = self
            .digit_tree_pipelines
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        if let Some(pipelines) = cached.as_ref() {
            return Ok(pipelines.clone());
        }
        let pipeline = |name: &'static str| {
            let function = self
                .digit_tree_library
                .get_function(name, None)
                .map_err(|message| MetalCommitError::FunctionLookup { name, message })?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| MetalCommitError::PipelineCompilation { name, message })?;
            if pipeline.max_total_threads_per_threadgroup() < THREADS as u64
                || pipeline.thread_execution_width() != 32
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "digit-tree kernels require 256-thread groups with 32-lane SIMD".into(),
                ));
            }
            Ok(pipeline)
        };
        let pipelines = Arc::new(DigitTreePipelines {
            partials: pipeline("akita_fp128_digit_tree_partials")?,
            reduce: pipeline("akita_fp128_digit_tree_reduce")?,
            bind: pipeline("akita_fp128_digit_tree_bind")?,
        });
        *cached = Some(pipelines.clone());
        Ok(pipelines)
    }

    pub(crate) fn begin_digit_tree(
        &self,
        pipelines: Arc<DigitTreePipelines>,
        explicit: &[F],
        padding: &[Fp128Limbs],
        polynomial: &[Fp128Limbs],
        domain_len: usize,
        arity: usize,
        degree: usize,
    ) -> Result<DigitTreeSession, MetalCommitError> {
        autoreleasepool(|| {
            let lanes = padding.len();
            let shape_valid = match arity {
                0 => lanes == 1 && Some(polynomial.len()) == degree.checked_add(1),
                2 | 4 => Some(lanes) == arity.checked_mul(polynomial.len()) && degree == arity,
                _ => false,
            };
            if !(1..=8).contains(&lanes)
                || !domain_len.is_power_of_two()
                || domain_len < 2
                || !shape_valid
                || degree > 4
                || polynomial.is_empty()
                || !explicit.len().is_multiple_of(lanes)
                || explicit.len() / lanes > domain_len
                || explicit.len() > u32::MAX as usize
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "invalid exact-prefix digit-tree geometry".into(),
                ));
            }
            let explicit_rows = explicit.len() / lanes;
            let pair_count = explicit_rows.div_ceil(2);
            let workgroups = pair_count.div_ceil(4 * THREADS).clamp(1, MAX_WORKGROUPS);
            let source_bytes = explicit
                .len()
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("digit-tree source buffer"))?
                .max(size_of_val(padding));
            let next_bytes = pair_count
                .checked_mul(lanes)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow("digit-tree bind buffer"))?
                .max(size_of_val(padding));
            let source = self.shared_buffer(source_bytes)?;
            // SAFETY: source owns at least explicit.len() aligned limb slots; each is initialized once.
            for (index, &value) in explicit.iter().enumerate() {
                unsafe {
                    source
                        .contents()
                        .cast::<Fp128Limbs>()
                        .add(index)
                        .write(Fp128Limbs::from_field(value));
                }
            }
            let next = self.shared_buffer(next_bytes)?;
            let padding_buffer = self.shared_buffer_from_slice(padding)?;
            let polynomial_buffer = self.shared_buffer_from_slice(polynomial)?;
            let partial_bytes = workgroups * COEFFICIENTS * size_of::<Fp128Limbs>();
            let output_bytes = COEFFICIENTS * size_of::<Fp128Limbs>();
            let partials = self.shared_buffer(partial_bytes)?;
            let output = self.shared_buffer(output_bytes)?;
            let allocation_bytes = source_bytes
                .checked_add(next_bytes)
                .and_then(|bytes| bytes.checked_add(size_of_val(padding)))
                .and_then(|bytes| bytes.checked_add(size_of_val(polynomial)))
                .and_then(|bytes| bytes.checked_add(partial_bytes))
                .and_then(|bytes| bytes.checked_add(output_bytes))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "digit-tree resident allocation",
                ))?;
            Ok(DigitTreeSession {
                pipelines,
                buffers: [source, next],
                current: 0,
                padding: padding_buffer,
                polynomial: polynomial_buffer,
                partials,
                output,
                params: DigitTreeParams {
                    explicit_rows: explicit_rows as u64,
                    pair_count: pair_count as u64,
                    lanes: lanes as u64,
                    first_len: 0,
                    parent_count: if arity == 0 {
                        0
                    } else {
                        polynomial.len() as u64
                    },
                    arity: arity as u64,
                    degree: degree as u64,
                    workgroups: workgroups as u64,
                },
                domain_len,
                allocation_bytes,
            })
        })
    }

    pub(crate) fn digit_tree_round(
        &self,
        session: &mut DigitTreeSession,
        first: &[Fp128Limbs],
        second: &[Fp128Limbs],
    ) -> Result<DigitTreeRoundOutcome, MetalCommitError> {
        if session.domain_len < 2
            || !first.len().is_power_of_two()
            || !second.len().is_power_of_two()
            || first.len().checked_mul(second.len()) != Some(session.domain_len / 2)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "digit-tree equality domain mismatch".into(),
            ));
        }
        if session.params.pair_count == 0 {
            return Ok(DigitTreeRoundOutcome {
                coefficients: [Fp128Limbs::default(); COEFFICIENTS],
                timings: DispatchTimings {
                    buffer_setup: Duration::ZERO,
                    command_wall: Duration::ZERO,
                    gpu: Some(Duration::ZERO),
                    readback_copy: Duration::ZERO,
                },
            });
        }
        autoreleasepool(|| {
            let setup_start = Instant::now();
            let first_buffer = self.shared_buffer_from_slice(first)?;
            let second_buffer = self.shared_buffer_from_slice(second)?;
            session.params.first_len = first.len() as u64;
            session.params.workgroups = usize::try_from(session.params.pair_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-tree pair count"))?
                .div_ceil(4 * THREADS)
                .clamp(1, MAX_WORKGROUPS) as u64;
            let buffer_setup = setup_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita class-indexed digit-tree q coefficients");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&session.pipelines.partials);
            for (index, buffer) in [
                &session.buffers[session.current],
                &session.padding,
                &session.polynomial,
                &first_buffer,
                &second_buffer,
                &session.partials,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            set_inline_bytes(encoder, 6, &session.params);
            encoder.dispatch_thread_groups(
                MTLSize::new(session.params.workgroups, 1, 1),
                MTLSize::new(THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&session.pipelines.reduce);
            encoder.set_buffer(0, Some(&session.partials), 0);
            encoder.set_buffer(1, Some(&session.output), 0);
            set_inline_bytes(encoder, 2, &session.params);
            encoder.dispatch_thread_groups(
                MTLSize::new(COEFFICIENTS as u64, 1, 1),
                MTLSize::new(THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed reduction initialized exactly five aligned limb values.
            let coefficients = unsafe {
                *session
                    .output
                    .contents()
                    .cast::<[Fp128Limbs; COEFFICIENTS]>()
            };
            Ok(DigitTreeRoundOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }

    pub(crate) fn digit_tree_bind(
        &self,
        session: &mut DigitTreeSession,
        challenge: Fp128Limbs,
    ) -> Result<DigitTreeBindOutcome, MetalCommitError> {
        if session.domain_len < 2 {
            return Err(MetalCommitError::UnsupportedShape(
                "cannot bind a final digit-tree row".into(),
            ));
        }
        autoreleasepool(|| {
            let next_rows = session.params.explicit_rows.div_ceil(2);
            let next_coefficients = next_rows.checked_mul(session.params.lanes).ok_or(
                MetalCommitError::ShapeOverflow("digit-tree next coefficient count"),
            )?;
            let mut command_wall = Duration::ZERO;
            let mut gpu = Some(Duration::ZERO);
            if next_coefficients != 0 {
                let command = self.queue.new_command_buffer();
                command.set_label("Akita class-indexed digit-tree bind");
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&session.pipelines.bind);
                encoder.set_buffer(0, Some(&session.buffers[session.current]), 0);
                encoder.set_buffer(1, Some(&session.padding), 0);
                encoder.set_buffer(2, Some(&session.buffers[1 - session.current]), 0);
                set_inline_bytes(encoder, 3, &session.params);
                set_inline_bytes(encoder, 4, &challenge);
                encoder.dispatch_threads(
                    MTLSize::new(next_coefficients, 1, 1),
                    MTLSize::new(THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                (command_wall, gpu) = complete_command(command)?;
            }
            session.current = 1 - session.current;
            session.params.explicit_rows = next_rows;
            session.params.pair_count = next_rows.div_ceil(2);
            session.domain_len /= 2;
            let readback_start = Instant::now();
            let final_value = if session.domain_len == 1 {
                let buffer = if next_rows == 0 {
                    &session.padding
                } else {
                    &session.buffers[session.current]
                };
                // SAFETY: the final explicit row was initialized by this completed bind, or padding by upload.
                Some(unsafe {
                    std::slice::from_raw_parts(
                        buffer.contents().cast::<Fp128Limbs>(),
                        session.params.lanes as usize,
                    )
                    .to_vec()
                })
            } else {
                None
            };
            Ok(DigitTreeBindOutcome {
                final_value,
                timings: DispatchTimings {
                    buffer_setup: Duration::ZERO,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }
}
