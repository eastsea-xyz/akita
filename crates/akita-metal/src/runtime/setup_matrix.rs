use super::*;
use akita_types::{public_matrix_page_xof_input, AkitaSetupSeed, PublicMatrixDerivation};

pub(crate) const SETUP_MATRIX_FIELDS_PER_PAGE: usize = 4096;
pub(crate) const SETUP_MATRIX_MAX_PAGES: usize = 256;
const SETUP_MATRIX_INPUT_BYTES: usize = 277;
pub(crate) const SETUP_MATRIX_DRAW_LIMIT: u32 = SETUP_MATRIX_FIELDS_PER_PAGE as u32 + 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct SetupMatrixParams {
    first_page_lo: u32,
    first_page_hi: u32,
    total_fields: u32,
    page_count: u32,
    draw_limit: u32,
    reserved: u32,
}

pub(crate) struct SetupMatrixDispatchOutcome {
    pub(crate) fields: Vec<F>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

impl MetalRuntime {
    pub(crate) fn dispatch_setup_matrix_with_limit(
        &self,
        seed: &AkitaSetupSeed,
        first_page: u64,
        field_count: usize,
        draw_limit: u32,
    ) -> Result<SetupMatrixDispatchOutcome, MetalCommitError> {
        if seed.derivation != PublicMatrixDerivation::Shake256PagedV1
            || seed.derivation.page_field_elements() != SETUP_MATRIX_FIELDS_PER_PAGE
            || field_count == 0
            || field_count > SETUP_MATRIX_MAX_PAGES * SETUP_MATRIX_FIELDS_PER_PAGE
            || draw_limit > SETUP_MATRIX_DRAW_LIMIT
        {
            return Err(MetalCommitError::UnsupportedShape(
                "paged setup matrix dispatch geometry".into(),
            ));
        }
        let page_count = field_count.div_ceil(SETUP_MATRIX_FIELDS_PER_PAGE);
        first_page
            .checked_add(page_count as u64 - 1)
            .ok_or(MetalCommitError::ShapeOverflow("setup matrix page range"))?;
        let message = public_matrix_page_xof_input::<F>(seed, 0);
        if message.len() != SETUP_MATRIX_INPUT_BYTES {
            return Err(MetalCommitError::UnsupportedShape(
                "paged setup matrix input encoding".into(),
            ));
        }
        let output_bytes = field_count
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow("setup matrix output bytes"))?;
        let status_bytes = page_count
            .checked_mul(size_of::<[u32; 2]>())
            .ok_or(MetalCommitError::ShapeOverflow("setup matrix status bytes"))?;
        self.validate_buffer_length(output_bytes)?;
        self.validate_buffer_length(status_bytes)?;
        let params = SetupMatrixParams {
            first_page_lo: first_page as u32,
            first_page_hi: (first_page >> 32) as u32,
            total_fields: u32::try_from(field_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("setup matrix field count"))?,
            page_count: page_count as u32,
            draw_limit,
            reserved: 0,
        };
        autoreleasepool(|| {
            let buffer_start = Instant::now();
            let input = self.shared_buffer_from_slice(&message)?;
            let output = self.shared_buffer(output_bytes)?;
            let status = self.shared_buffer(status_bytes)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita canonical public setup matrix SHAKE256");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.setup_matrix_pipeline);
            encoder.set_buffer(0, Some(&input), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&status), 0);
            set_inline_bytes(encoder, 3, &params);
            let threads = self
                .setup_matrix_pipeline
                .thread_execution_width()
                .min(
                    self.setup_matrix_pipeline
                        .max_total_threads_per_threadgroup(),
                )
                .max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new((page_count as u64).div_ceil(threads), 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: every dispatched page writes one status after completion.
            let statuses = unsafe {
                std::slice::from_raw_parts(status.contents().cast::<[u32; 2]>(), page_count)
            };
            for (page, counts) in statuses.iter().enumerate() {
                let wanted = (field_count - page * SETUP_MATRIX_FIELDS_PER_PAGE)
                    .min(SETUP_MATRIX_FIELDS_PER_PAGE);
                if counts[0] as usize != wanted || counts[1] > draw_limit {
                    return Err(MetalCommitError::UnsupportedShape(format!(
                        "setup matrix page {} rejection budget exhausted (accepted {}, draws {})",
                        first_page + page as u64,
                        counts[0],
                        counts[1]
                    )));
                }
            }
            // SAFETY: the accepted counts prove all requested outputs were written;
            // the command is complete and owns the shared buffer until decoding ends.
            let limbs = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), field_count)
            };
            let mut fields = Vec::new();
            fields.try_reserve_exact(field_count).map_err(|error| {
                MetalCommitError::UnsupportedShape(format!(
                    "setup matrix readback allocation: {error}"
                ))
            })?;
            for (index, &value) in limbs.iter().enumerate() {
                fields.push(value.into_field(index)?);
            }
            Ok(SetupMatrixDispatchOutcome {
                fields,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes: output_bytes + status_bytes + message.len(),
            })
        })
    }

    #[cfg(test)]
    pub(crate) fn setup_matrix_sampling_oracle(
        &self,
        candidates: &[Fp128Limbs],
        wanted: usize,
    ) -> Result<(Vec<F>, [u32; 2]), MetalCommitError> {
        if candidates.is_empty() || wanted == 0 || wanted > candidates.len() {
            return Err(MetalCommitError::UnsupportedShape(
                "setup matrix scripted stream".into(),
            ));
        }
        let params = [candidates.len() as u32, wanted as u32];
        autoreleasepool(|| {
            let input = self.shared_buffer_from_slice(candidates)?;
            let output = self.shared_buffer(wanted * size_of::<Fp128Limbs>())?;
            let status = self.shared_buffer(size_of::<[u32; 2]>())?;
            let command = self.queue.new_command_buffer();
            command.set_label("Akita setup matrix rejection sampling oracle");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.setup_matrix_sampling_pipeline);
            encoder.set_buffer(0, Some(&input), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&status), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            let _ = complete_command(command)?;
            // SAFETY: the single completed thread always writes its status.
            let counts = unsafe { status.contents().cast::<[u32; 2]>().read() };
            if counts[0] as usize != wanted {
                return Err(MetalCommitError::UnsupportedShape(
                    "scripted sampling exhausted".into(),
                ));
            }
            // SAFETY: the completed thread's accepted count covers all output slots.
            let values = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), wanted)
            };
            let fields = values
                .iter()
                .enumerate()
                .map(|(i, v)| v.into_field(i))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((fields, counts))
        })
    }
}
