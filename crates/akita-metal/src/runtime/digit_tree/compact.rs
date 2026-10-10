use super::*;
use akita_prover::protocol::sumcheck::digit_range::{
    ClassIndexedDigitTreeCompactSecondRoundInput, ClassIndexedDigitTreeCompactSourceInput,
};
use std::mem::align_of;

pub(crate) struct CompactDigitTreeSource {
    indices: Buffer,
    pub(crate) domain_len: usize,
    live_len: usize,
    pair_lut_count: usize,
    pub(crate) allocation_bytes: usize,
}

pub(super) struct CompactDigitTreeState {
    source: Arc<CompactDigitTreeSource>,
    coefficients: Buffer,
    folded_pairs: Option<Buffer>,
    phase: u64,
    use_quartet_lut: bool,
}

#[repr(C)]
struct CompactDigitTreeParams {
    table: DigitTreeParams,
    pair_lut_count: u64,
    phase: u64,
    use_quartet_lut: u64,
}

// Rust and Metal both lay out eight u64 table fields followed by three u64 compact fields.
const _: [(); 64] = [(); size_of::<DigitTreeParams>()];
const _: [(); 8] = [(); align_of::<DigitTreeParams>()];
const _: [(); 88] = [(); size_of::<CompactDigitTreeParams>()];
const _: [(); 8] = [(); align_of::<CompactDigitTreeParams>()];

#[test]
fn compact_digit_tree_params_match_metal_abi() {
    assert_eq!(size_of::<DigitTreeParams>(), 64);
    assert_eq!(align_of::<DigitTreeParams>(), 8);
    assert_eq!(size_of::<CompactDigitTreeParams>(), 88);
    assert_eq!(align_of::<CompactDigitTreeParams>(), 8);
    assert_eq!(
        std::mem::offset_of!(CompactDigitTreeParams, pair_lut_count),
        64
    );
    assert_eq!(std::mem::offset_of!(CompactDigitTreeParams, phase), 72);
    assert_eq!(
        std::mem::offset_of!(CompactDigitTreeParams, use_quartet_lut),
        80
    );
}

impl MetalRuntime {
    pub(crate) fn prepare_compact_digit_tree_source(
        &self,
        input: ClassIndexedDigitTreeCompactSourceInput<'_>,
    ) -> Result<CompactDigitTreeSource, MetalCommitError> {
        let pair_lut_count = input.class_count.checked_mul(input.class_count).ok_or(
            MetalCommitError::ShapeOverflow("compact digit-tree class pairs"),
        )?;
        if input.domain_len < 8
            || !input.domain_len.is_power_of_two()
            || input.live_len > input.domain_len
            || !matches!(input.class_count, 8 | 16 | 32)
            || input.ordered_pairs.len() != input.live_len.div_ceil(2)
            || input
                .ordered_pairs
                .iter()
                .any(|&index| usize::from(index) >= pair_lut_count)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "invalid compact digit-tree source".into(),
            ));
        }
        if !input.live_len.is_multiple_of(2)
            && input
                .ordered_pairs
                .last()
                .is_some_and(|&index| usize::from(index) % input.class_count != 0)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "odd compact prefix must pad its last right class with zero".into(),
            ));
        }
        let indices = if input.ordered_pairs.is_empty() {
            self.shared_buffer_from_slice(&[0u16])?
        } else {
            self.shared_buffer_from_slice(input.ordered_pairs)?
        };
        Ok(CompactDigitTreeSource {
            indices,
            domain_len: input.domain_len,
            live_len: input.live_len,
            pair_lut_count,
            allocation_bytes: size_of_val(input.ordered_pairs).max(size_of::<u16>()),
        })
    }

    pub(crate) fn attach_compact_digit_tree(
        &self,
        session: &mut DigitTreeSession,
        source: Arc<CompactDigitTreeSource>,
        coefficients: &[[F; 5]],
    ) -> Result<usize, MetalCommitError> {
        if coefficients.len() != source.pair_lut_count {
            return Err(MetalCommitError::UnsupportedShape(
                "compact pair LUT length mismatch".into(),
            ));
        }
        let coefficients = coefficients
            .as_flattened()
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let coefficient_buffer = self.shared_buffer_from_slice(&coefficients)?;
        let quarter_rows = source.live_len.div_ceil(4);
        let field_bytes = quarter_rows
            .checked_mul(session.params.lanes as usize)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "compact digit-tree materialization",
            ))?
            .max(session.params.lanes as usize * size_of::<Fp128Limbs>());
        let next_bytes = quarter_rows
            .div_ceil(2)
            .checked_mul(session.params.lanes as usize)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "compact digit-tree ping-pong",
            ))?
            .max(session.params.lanes as usize * size_of::<Fp128Limbs>());
        let workgroups = source
            .live_len
            .div_ceil(2)
            .div_ceil(4 * THREADS)
            .clamp(1, MAX_WORKGROUPS);
        let partial_bytes = workgroups * COEFFICIENTS * size_of::<Fp128Limbs>();
        let extra = field_bytes
            .checked_add(next_bytes)
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(coefficients.as_slice())))
            .ok_or(MetalCommitError::ShapeOverflow(
                "compact digit-tree allocation",
            ))?;
        session.buffers = [
            self.shared_buffer(field_bytes)?,
            self.shared_buffer(next_bytes)?,
        ];
        session.partials = self.shared_buffer(partial_bytes)?;
        session.domain_len = source.domain_len;
        session.params.explicit_rows = source.live_len as u64;
        session.params.pair_count = source.live_len.div_ceil(2) as u64;
        session.params.workgroups = workgroups as u64;
        session.allocation_bytes = session.allocation_bytes.saturating_add(extra);
        session.compact = Some(CompactDigitTreeState {
            source,
            coefficients: coefficient_buffer,
            folded_pairs: None,
            phase: 0,
            use_quartet_lut: false,
        });
        Ok(extra)
    }

    pub(crate) fn prepare_compact_digit_tree_second_round(
        &self,
        session: &mut DigitTreeSession,
        input: ClassIndexedDigitTreeCompactSecondRoundInput<'_, F>,
    ) -> Result<usize, MetalCommitError> {
        let compact = session.compact.as_mut().ok_or_else(|| {
            MetalCommitError::UnsupportedShape("compact digit-tree is already materialized".into())
        })?;
        let expected = compact
            .source
            .pair_lut_count
            .checked_mul(session.params.lanes as usize)
            .ok_or(MetalCommitError::ShapeOverflow("compact folded pair LUT"))?;
        if compact.phase != 1 || input.folded_pairs.len() != expected {
            return Err(MetalCommitError::UnsupportedShape(
                "compact folded pair LUT mismatch".into(),
            ));
        }
        if let Some(lut) = input.quartet_coefficients {
            if compact.source.pair_lut_count > 64
                || lut.len() != compact.source.pair_lut_count * compact.source.pair_lut_count
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "compact quartet LUT must be bounded".into(),
                ));
            }
        }
        let folded = input
            .folded_pairs
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        compact.folded_pairs = Some(self.shared_buffer_from_slice(&folded)?);
        let mut extra = size_of_val(folded.as_slice());
        compact.coefficients = if let Some(lut) = input.quartet_coefficients {
            let limbs = lut
                .as_flattened()
                .iter()
                .copied()
                .map(Fp128Limbs::from_field)
                .collect::<Vec<_>>();
            extra = extra.checked_add(size_of_val(limbs.as_slice())).ok_or(
                MetalCommitError::ShapeOverflow("compact quartet LUT allocation"),
            )?;
            compact.use_quartet_lut = true;
            self.shared_buffer_from_slice(&limbs)?
        } else {
            session.padding.clone()
        };
        session.allocation_bytes = session.allocation_bytes.saturating_add(extra);
        Ok(extra)
    }

    pub(super) fn compact_digit_tree_round(
        &self,
        session: &mut DigitTreeSession,
        first: &[Fp128Limbs],
        second: &[Fp128Limbs],
    ) -> Result<DigitTreeRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let setup = Instant::now();
            let compact = session.compact.as_ref().ok_or_else(|| {
                MetalCommitError::UnsupportedShape("missing compact source".into())
            })?;
            if compact.phase == 1 && compact.folded_pairs.is_none() {
                return Err(MetalCommitError::UnsupportedShape(
                    "second-round LUT not installed".into(),
                ));
            }
            let first_buffer = self.shared_buffer_from_slice(first)?;
            let second_buffer = self.shared_buffer_from_slice(second)?;
            session.params.first_len = first.len() as u64;
            session.params.workgroups = (session.params.pair_count as usize)
                .div_ceil(4 * THREADS)
                .clamp(1, MAX_WORKGROUPS) as u64;
            let params = CompactDigitTreeParams {
                table: session.params,
                pair_lut_count: compact.source.pair_lut_count as u64,
                phase: compact.phase,
                use_quartet_lut: u64::from(compact.use_quartet_lut),
            };
            let buffer_setup = setup.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita compact digit-tree weighted LUT coefficients");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&session.pipelines.compact_partials);
            let folded = compact.folded_pairs.as_ref().unwrap_or(&session.padding);
            for (index, buffer) in [
                &compact.source.indices,
                folded,
                &session.padding,
                &session.polynomial,
                &compact.coefficients,
                &first_buffer,
                &second_buffer,
                &session.partials,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            set_inline_bytes(encoder, 8, &params);
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
            let readback = Instant::now();
            // SAFETY: the completed reduction initialized exactly five canonical limb slots.
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
                    readback_copy: readback.elapsed(),
                },
            })
        })
    }

    pub(super) fn compact_digit_tree_bind(
        &self,
        session: &mut DigitTreeSession,
        challenge: Fp128Limbs,
    ) -> Result<DigitTreeBindOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let compact = session.compact.as_ref().ok_or_else(|| {
                MetalCommitError::UnsupportedShape("missing compact source".into())
            })?;
            let mut command_wall = Duration::ZERO;
            let mut gpu = Some(Duration::ZERO);
            let next_rows = session.params.explicit_rows.div_ceil(2);
            if compact.phase == 1 {
                let folded = compact.folded_pairs.as_ref().ok_or_else(|| {
                    MetalCommitError::UnsupportedShape("second-round LUT not installed".into())
                })?;
                let params = CompactDigitTreeParams {
                    table: session.params,
                    pair_lut_count: compact.source.pair_lut_count as u64,
                    phase: 1,
                    use_quartet_lut: u64::from(compact.use_quartet_lut),
                };
                let count = next_rows.checked_mul(session.params.lanes).ok_or(
                    MetalCommitError::ShapeOverflow("compact materialization lanes"),
                )?;
                if count > 0 {
                    let command = self.queue.new_command_buffer();
                    command.set_label("Akita compact quartet to resident field rows");
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&session.pipelines.compact_materialize);
                    for (index, buffer) in [&compact.source.indices, folded, &session.buffers[0]]
                        .into_iter()
                        .enumerate()
                    {
                        encoder.set_buffer(index as u64, Some(buffer), 0);
                    }
                    set_inline_bytes(encoder, 3, &params);
                    set_inline_bytes(encoder, 4, &challenge);
                    encoder.dispatch_threads(
                        MTLSize::new(count, 1, 1),
                        MTLSize::new(THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    (command_wall, gpu) = complete_command(command)?;
                }
                session.current = 0;
                session.compact = None;
            } else {
                session
                    .compact
                    .as_mut()
                    .ok_or_else(|| {
                        MetalCommitError::UnsupportedShape("missing compact source".into())
                    })?
                    .phase = 1;
            }
            session.params.explicit_rows = next_rows;
            session.params.pair_count = next_rows.div_ceil(2);
            session.domain_len /= 2;
            Ok(DigitTreeBindOutcome {
                final_value: None,
                timings: DispatchTimings {
                    buffer_setup: Duration::ZERO,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
            })
        })
    }
}
