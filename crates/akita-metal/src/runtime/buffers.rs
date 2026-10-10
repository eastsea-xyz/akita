use super::*;
use crate::field::MetalField;
use metal::CommandBuffer;

impl MetalRuntime {
    pub(crate) fn shared_buffer_from_slice<T>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    pub(super) fn shared_buffer_from_digit_rows<const D: usize>(
        &self,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = digit_vectors.iter().try_fold(0usize, |total, digits| {
            total
                .checked_add(size_of_val(*digits))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row input bytes"))
        })?;
        let buffer = self.shared_buffer(bytes)?;
        let mut destination = buffer.contents().cast::<u8>();
        for digits in digit_vectors {
            let len = size_of_val(*digits);
            // SAFETY: `buffer` owns `bytes`, the checked sum of the disjoint
            // source lengths, and both pointers are valid for `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(digits.as_ptr().cast::<u8>(), destination, len);
                destination = destination.add(len);
            }
        }
        Ok(buffer)
    }

    pub(crate) fn shared_slice_buffer<'a, T>(
        &self,
        values: &'a [T],
    ) -> Result<SharedSliceBuffer<'a, T>, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = values
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast_mut().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(values)?
        };
        Ok(SharedSliceBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(super) fn shared_byte_buffer_from_slices<'a>(
        &self,
        values: &[&'a [i8]],
    ) -> Result<SharedByteBuffer<'a>, MetalCommitError> {
        let bytes = values.iter().try_fold(0usize, |total, value| {
            total
                .checked_add(size_of_val(*value))
                .ok_or(MetalCommitError::ShapeOverflow("byte input"))
        })?;
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal byte input".into(),
            ));
        }
        let zero_copy = values.len() == 1
            && values[0]
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values[0].as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            let buffer = self.shared_buffer(bytes)?;
            let mut destination = buffer.contents().cast::<u8>();
            for value in values {
                let len = size_of_val(*value);
                // SAFETY: `buffer` owns the checked sum of every source length;
                // the destination advances over disjoint initialized ranges.
                unsafe {
                    std::ptr::copy_nonoverlapping(value.as_ptr().cast::<u8>(), destination, len);
                    destination = destination.add(len);
                }
            }
            buffer
        };
        Ok(SharedByteBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    /// Encode directly into one owned CPU/GPU shared allocation.
    pub(crate) fn shared_buffer_from_fields<T: MetalField>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = values
            .len()
            .checked_mul(size_of::<T::DeviceElement>())
            .ok_or(MetalCommitError::ShapeOverflow("field input bytes"))?;
        let buffer = self.shared_buffer(bytes)?;
        let destination = buffer.contents().cast::<T::DeviceElement>();
        for (index, &value) in values.iter().enumerate() {
            // SAFETY: `buffer` owns the checked byte length for all device
            // elements, and each write initializes one disjoint element before
            // the buffer is returned or submitted to a GPU command. Encoding
            // uses MetalField rather than assuming the host field layout;
            // unaligned writes avoid an additional host alignment requirement.
            unsafe {
                destination.add(index).write_unaligned(value.into_device());
            }
        }
        Ok(buffer)
    }

    pub(super) fn packed_lane_buffer<'a>(
        &self,
        lanes: &'a [u8],
    ) -> Result<PackedLaneBuffer<'a>, MetalCommitError> {
        let bytes = lanes.len();
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = lanes
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                lanes.as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(lanes)?
        };
        Ok(PackedLaneBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(super) fn packed_lane_uploads<'a, S: PackedLaneSource>(
        &self,
        source: &'a S,
        live_rows: u64,
        lane_stride: u64,
        window_capacity: usize,
    ) -> Result<PackedLaneUploads<'a>, MetalCommitError> {
        let prefix_bytes = live_rows
            .checked_mul(lane_stride)
            .ok_or(MetalCommitError::ShapeOverflow("packed unique lane bytes"))?;
        let unique_lane_bytes = usize::try_from(prefix_bytes)
            .map_err(|_| MetalCommitError::ShapeOverflow("packed unique lane bytes"))?;
        let mut stats = PackedLaneUploadStats::new(unique_lane_bytes);
        let resident = if resident_prefix_fits(
            prefix_bytes,
            self.device.max_buffer_length(),
            self.device.current_allocated_size(),
            self.device.recommended_max_working_set_size(),
        ) {
            let rows = usize::try_from(live_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("packed live rows"))?;
            let stride = usize::try_from(lane_stride)
                .map_err(|_| MetalCommitError::ShapeOverflow("packed lane stride"))?;
            let lanes = source.wait_lanes(0..rows, stride)?;
            let buffer = self.packed_lane_buffer(lanes)?;
            buffer.buffer.set_label("Akita packed lane live prefix");
            stats.record_buffer(&buffer)?;
            stats.resident_prefix_reused = true;
            Some(buffer)
        } else {
            None
        };
        Ok(PackedLaneUploads {
            resident,
            windows: Vec::with_capacity(if stats.resident_prefix_reused {
                0
            } else {
                window_capacity
            }),
            stats,
        })
    }

    pub(super) fn shared_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal output buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared))
    }

    pub(super) fn private_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal scratch buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate))
    }

    pub(super) fn validate_buffer_length(&self, bytes: usize) -> Result<(), MetalCommitError> {
        let requested =
            u64::try_from(bytes).map_err(|_| MetalCommitError::ShapeOverflow("buffer bytes"))?;
        let maximum = self.device.max_buffer_length();
        if requested > maximum {
            return Err(MetalCommitError::BufferTooLong { requested, maximum });
        }
        Ok(())
    }
}

/// Drain submitted work before externally backed lane storage can drop.
pub(super) struct PackedSubmissionGuard {
    commands: Vec<CommandBuffer>,
}

impl PackedSubmissionGuard {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn record(&mut self, command: &CommandBufferRef) {
        // Capacity is reserved before any submission, so tracking cannot
        // allocate after commit or leave earlier GPU readers unprotected.
        debug_assert!(self.commands.len() < self.commands.capacity());
        self.commands.push(command.to_owned());
    }
}

impl Drop for PackedSubmissionGuard {
    fn drop(&mut self) {
        for command in &self.commands {
            if !matches!(
                command.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            ) {
                command.wait_until_completed();
            }
        }
    }
}

/// Logical input lengths; shared/no-copy views can overlap physical pages.
pub(super) struct PackedLaneUploadStats {
    pub(super) input_zero_copy: bool,
    pub(super) resident_prefix_reused: bool,
    unique_lane_bytes: usize,
    window_lane_bytes: usize,
    registered_lane_bytes: usize,
    uploaded_lane_bytes: usize,
    input_buffers: usize,
}

impl PackedLaneUploadStats {
    pub(super) fn new(unique_lane_bytes: usize) -> Self {
        Self {
            input_zero_copy: true,
            resident_prefix_reused: false,
            unique_lane_bytes,
            window_lane_bytes: 0,
            registered_lane_bytes: 0,
            uploaded_lane_bytes: 0,
            input_buffers: 0,
        }
    }

    pub(super) fn record_window(&mut self, bytes: usize) -> Result<(), MetalCommitError> {
        self.window_lane_bytes = self
            .window_lane_bytes
            .checked_add(bytes)
            .ok_or(MetalCommitError::ShapeOverflow("packed window lane bytes"))?;
        Ok(())
    }

    pub(super) fn record_buffer(
        &mut self,
        buffer: &PackedLaneBuffer<'_>,
    ) -> Result<(), MetalCommitError> {
        let bytes = usize::try_from(buffer.buffer.length())
            .map_err(|_| MetalCommitError::ShapeOverflow("packed registered lane bytes"))?;
        self.registered_lane_bytes = self.registered_lane_bytes.checked_add(bytes).ok_or(
            MetalCommitError::ShapeOverflow("packed registered lane bytes"),
        )?;
        if !buffer.zero_copy {
            self.uploaded_lane_bytes = self
                .uploaded_lane_bytes
                .checked_add(bytes)
                .ok_or(MetalCommitError::ShapeOverflow("packed copied lane bytes"))?;
        }
        self.input_buffers = self
            .input_buffers
            .checked_add(1)
            .ok_or(MetalCommitError::ShapeOverflow("packed input buffer count"))?;
        self.input_zero_copy &= buffer.zero_copy;
        Ok(())
    }

    pub(super) fn log(&self, params: &PackedOneHotCommitParams) {
        tracing::debug!(
            target: "akita_metal::packed_lane_upload",
            ring_d = params.ring_d,
            onehot_k = params.onehot_k,
            num_rows = params.num_rows,
            num_columns = params.num_columns,
            positions_per_block = params.positions_per_block,
            blocks_per_column = params.blocks_per_column,
            full_blocks_per_column = params.full_blocks_per_column,
            input_zero_copy = self.input_zero_copy,
            resident_prefix_reused = self.resident_prefix_reused,
            unique_lane_bytes = self.unique_lane_bytes,
            window_lane_bytes = self.window_lane_bytes,
            registered_lane_bytes = self.registered_lane_bytes,
            uploaded_lane_bytes = self.uploaded_lane_bytes,
            input_buffers = self.input_buffers,
            "packed Metal lane upload accounting"
        );
    }
}

fn resident_prefix_fits(bytes: u64, maximum_buffer: u64, allocated: u64, recommended: u64) -> bool {
    bytes != 0
        && bytes <= maximum_buffer
        && allocated
            .checked_add(bytes)
            .is_some_and(|total| total <= recommended)
}

/// Keep input owners live until the final reduction has completed.
pub(super) struct PackedLaneUploads<'a> {
    resident: Option<PackedLaneBuffer<'a>>,
    windows: Vec<PackedLaneBuffer<'a>>,
    pub(super) stats: PackedLaneUploadStats,
}

impl<'a> PackedLaneUploads<'a> {
    pub(super) fn buffer_for_window(
        &mut self,
        runtime: &MetalRuntime,
        lanes: &'a [u8],
        byte_offset: usize,
    ) -> Result<(&Buffer, u64), MetalCommitError> {
        self.stats.record_window(lanes.len())?;
        if let Some(resident) = &self.resident {
            let end =
                byte_offset
                    .checked_add(lanes.len())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "packed resident lane range",
                    ))?;
            if u64::try_from(end)
                .ok()
                .is_none_or(|end| end > resident.buffer.length())
            {
                return Err(MetalCommitError::ShapeOverflow(
                    "packed resident lane range",
                ));
            }
            let offset = u64::try_from(byte_offset)
                .map_err(|_| MetalCommitError::ShapeOverflow("packed resident lane offset"))?;
            return Ok((&resident.buffer, offset));
        }
        let buffer = runtime.packed_lane_buffer(lanes)?;
        self.stats.record_buffer(&buffer)?;
        self.windows.push(buffer);
        Ok((
            &self
                .windows
                .last()
                .ok_or(MetalCommitError::ShapeOverflow("packed input buffer count"))?
                .buffer,
            0,
        ))
    }
}

#[cfg(test)]
mod lane_residency_tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{self, RecvTimeoutError, Sender};
    use std::sync::{Arc, Mutex};

    use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, OneHotPoly, RootCommitSource,
    };
    use akita_types::{RingVec, SetupMatrixCapacity};
    use tracing::{span, Event, Metadata, Subscriber};

    use super::*;

    struct CapturedFields(BTreeMap<String, String>);

    impl tracing::field::Visit for CapturedFields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    struct UploadCapture(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

    impl Subscriber for UploadCapture {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.target() == "akita_metal::packed_lane_upload"
        }

        fn new_span(&self, _span: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            let mut fields = CapturedFields(BTreeMap::new());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
    }

    struct FailingLaneSource<'a> {
        lanes: &'a [u8],
        reads: AtomicUsize,
        reached_failure: Sender<()>,
    }

    impl PackedLaneSource for FailingLaneSource<'_> {
        fn lane_count(&self) -> usize {
            self.lanes.len()
        }

        fn wait_lanes(
            &self,
            rows: Range<usize>,
            lane_stride: usize,
        ) -> Result<&[u8], MetalCommitError> {
            if self.reads.fetch_add(1, Ordering::SeqCst) == 2 {
                let _ = self.reached_failure.send(());
                return Err(MetalCommitError::ShapeOverflow(
                    "injected later lane failure",
                ));
            }
            self.lanes
                .get(rows.start * lane_stride..rows.end * lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("test lane range"))
        }
    }

    #[test]
    fn later_lane_failure_drains_submitted_gpu_reads() {
        const D: usize = 512;
        const K: usize = 16;
        const COLUMNS: usize = 61;
        const POSITIONS: usize = 64;
        const BLOCKS: usize = 8;
        const ROWS: usize = BLOCKS * POSITIONS * (D / K);
        const BYTES: usize = ROWS * COLUMNS;
        let owner = vec![1u8; BYTES + PACKED_ONEHOT_BUFFER_ALIGNMENT];
        let offset = (PACKED_ONEHOT_BUFFER_ALIGNMENT
            - owner.as_ptr().addr() % PACKED_ONEHOT_BUFFER_ALIGNMENT)
            % PACKED_ONEHOT_BUFFER_ALIGNMENT;
        let lanes = &owner[offset..offset + BYTES];
        assert!(lanes
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT));
        assert!(lanes.len().is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT));
        let runtime = MetalRuntime::new().unwrap();
        let matrix = runtime
            .shared_buffer_from_fields(&vec![F::from_u64(0); POSITIONS * D])
            .unwrap();
        let params = PackedOneHotCommitParams {
            num_rows: ROWS as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            column_capacity: 64,
            onehot_k: K as u64,
            ring_d: D as u64,
            n_a: 1,
            positions_per_block: POSITIONS as u64,
            num_digits_inner: 1,
            blocks_per_column: BLOCKS as u64,
            full_blocks_per_column: BLOCKS as u64,
            boundary_columns: 0,
            num_blocks: (COLUMNS * BLOCKS) as u64,
            task_offset: 0,
            dispatch_tasks: (COLUMNS * BLOCKS) as u64,
            lane_row_offset: 0,
            output_coefficients: (64 * BLOCKS * D) as u64,
            columns_per_threadgroup: 1,
            position_partials_per_block: FP128_D512_POSITION_PARTIALS as u64,
            positions_per_partial: (POSITIONS / FP128_D512_POSITION_PARTIALS) as u64,
            log_ring_d: D.trailing_zeros() as u64,
            zero_column_mask: 0,
        };
        let (failure_tx, failure_rx) = mpsc::channel();
        let source = FailingLaneSource {
            lanes,
            reads: AtomicUsize::new(0),
            reached_failure: failure_tx,
        };
        let (done_tx, done_rx) = mpsc::channel();
        let event = runtime.device.new_shared_event();
        let blocker = autoreleasepool(|| runtime.queue.new_command_buffer().to_owned());
        blocker.encode_wait_for_event(&event, 1);
        blocker.commit();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                let result =
                    runtime.dispatch_packed_onehot_source(&matrix, &source, &[], params, 1);
                let _ = done_tx.send(());
                result
            });
            let reached = failure_rx.recv_timeout(Duration::from_secs(5));
            let early_return = done_rx.recv_timeout(Duration::from_millis(100));
            // Release the queue before any assertion or join can unwind.
            event.set_signaled_value(1);
            blocker.wait_until_completed();
            let result = worker.join().unwrap();
            assert!(reached.is_ok(), "the later source failure was not reached");
            assert!(
                matches!(early_return, Err(RecvTimeoutError::Timeout)),
                "dispatch returned while a submitted GPU read was still blocked"
            );
            assert!(matches!(
                result,
                Err(MetalCommitError::ShapeOverflow(
                    "injected later lane failure"
                ))
            ));
        });
    }

    #[test]
    fn combined_lane_prefix_requires_buffer_and_working_set_capacity() {
        assert!(resident_prefix_fits(32, 32, 48, 80));
        assert!(!resident_prefix_fits(33, 32, 0, 80));
        assert!(!resident_prefix_fits(32, 64, 49, 80));
        assert!(!resident_prefix_fits(1, 64, u64::MAX, u64::MAX));
        assert!(!resident_prefix_fits(0, 64, 0, 80));
    }

    #[test]
    fn unaligned_packed_lanes_upload_once_and_preserve_cpu_commitment() {
        const D: usize = 512;
        const K: usize = 16;
        const COLUMNS: usize = 17;
        const CAPACITY: usize = 64;
        const POSITIONS: usize = 64;
        const BLOCKS: usize = 4;
        const LIVE_BLOCKS: usize = 3;
        const ROWS_PER_BLOCK: usize = POSITIONS * (D / K);
        const ROWS: usize = BLOCKS * ROWS_PER_BLOCK;
        const LIVE_ROWS: usize = LIVE_BLOCKS * ROWS_PER_BLOCK;
        const ZERO_MASK: u64 = 0b1010;
        let mut owner = vec![0u8; ROWS * COLUMNS + 1];
        let offset = usize::from(
            !owner
                .as_ptr()
                .addr()
                .wrapping_add(1)
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT),
        );
        let lanes = &mut owner[offset..offset + ROWS * COLUMNS];
        assert!(!lanes
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT));
        for (index, lane) in lanes.iter_mut().enumerate() {
            if index / COLUMNS < LIVE_ROWS && index.is_multiple_of(97) {
                *lane = ((index.wrapping_mul(73) % (K - 1)) + 1) as u8;
            }
        }
        let mut active_zero_rows = vec![0u64; ROWS / u64::BITS as usize];
        for row in (0..LIVE_ROWS).step_by(11) {
            active_zero_rows[row / u64::BITS as usize] |= 1u64 << (row % u64::BITS as usize);
        }
        let indices = (0..CAPACITY)
            .flat_map(|column| {
                let lanes = &*lanes;
                let active_zero_rows = &active_zero_rows;
                (0..ROWS).map(move |row| {
                    if column >= COLUMNS {
                        None
                    } else {
                        let lane = lanes[row * COLUMNS + column];
                        let committed_zero = ZERO_MASK & (1u64 << column) != 0
                            && active_zero_rows[row / u64::BITS as usize]
                                & (1u64 << (row % u64::BITS as usize))
                                != 0;
                        (lane != 0 || committed_zero).then_some(lane)
                    }
                })
            })
            .collect();
        let polynomial = OneHotPoly::<F, u8>::new(K, indices).unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            (ROWS * CAPACITY * K).trailing_zeros() as usize,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS * D,
            },
        )
        .unwrap();
        let cpu = CpuBackend::DEFAULT;
        let prepared = cpu.prepare_setup(&setup).unwrap();
        let view = <OneHotPoly<F, u8> as RootCommitSource<F, D>>::commit_view(&polynomial).unwrap();
        let expected = cpu
            .commit_inner_group(
                &prepared,
                vec![view],
                CommitInnerPlan {
                    n_a: 1,
                    num_positions_per_block: POSITIONS,
                    num_digits_inner: 1,
                    log_basis_inner: 3,
                },
            )
            .unwrap()
            .remove(0);
        let runtime = MetalRuntime::new().unwrap();
        let matrix = runtime
            .shared_buffer_from_fields(setup.expanded.shared_matrix.as_field_slice())
            .unwrap();
        let params = PackedOneHotCommitParams {
            num_rows: ROWS as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            column_capacity: CAPACITY as u64,
            onehot_k: K as u64,
            ring_d: D as u64,
            n_a: 1,
            positions_per_block: POSITIONS as u64,
            num_digits_inner: 1,
            blocks_per_column: BLOCKS as u64,
            full_blocks_per_column: LIVE_BLOCKS as u64,
            boundary_columns: 0,
            num_blocks: (COLUMNS * LIVE_BLOCKS) as u64,
            task_offset: 0,
            dispatch_tasks: (COLUMNS * LIVE_BLOCKS) as u64,
            lane_row_offset: 0,
            output_coefficients: (CAPACITY * BLOCKS * D) as u64,
            columns_per_threadgroup: 1,
            position_partials_per_block: FP128_D512_POSITION_PARTIALS as u64,
            positions_per_partial: (POSITIONS / FP128_D512_POSITION_PARTIALS) as u64,
            log_ring_d: D.trailing_zeros() as u64,
            zero_column_mask: ZERO_MASK,
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let outcome = tracing::subscriber::with_default(UploadCapture(events.clone()), || {
            runtime.dispatch_packed_onehot(&matrix, lanes, &active_zero_rows, params, 1)
        })
        .unwrap();
        assert!(outcome.command_buffers > 2);
        assert!(!outcome.input_zero_copy);
        let fields = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, limbs)| F::from_device(limbs, index).unwrap())
            .collect();
        let actual = RingVec::from_coeffs_with_ring_dim(fields, D).unwrap();
        assert_eq!(expected.inner_rows, actual);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event["input_buffers"], "1");
        assert_eq!(event["resident_prefix_reused"], "true");
        assert_eq!(event["input_zero_copy"], "false");
        for key in [
            "unique_lane_bytes",
            "registered_lane_bytes",
            "uploaded_lane_bytes",
        ] {
            assert_eq!(event[key].parse::<usize>().unwrap(), LIVE_ROWS * COLUMNS);
        }
        assert!(event["window_lane_bytes"].parse::<usize>().unwrap() > LIVE_ROWS * COLUMNS);
    }
}
