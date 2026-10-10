use super::*;

/// Physical ordinary-2D layout, independent of the logical row/column view.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixSpectrumLayout {
    pub(crate) prime_stride: u64,
    pub(crate) ring_stride: u64,
    pub(crate) frequency_offset: u64,
}

/// Explicit ordinary-2D contract; it never guesses legacy D storage from size.
pub(crate) struct RecursiveMatrixView<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) layout: MatrixSpectrumLayout,
}

impl<'a> RecursiveMatrixView<'a> {
    /// Only ordinary zero-padded 2D DIF producers may call this constructor.
    /// Upper D entries equal the twisted-D spectrum under the same order-2D root.
    pub(crate) fn from_ordinary_buffer<const D: usize>(
        buffer: &'a Buffer,
    ) -> Result<Self, MetalCommitError> {
        let ring_stride = D.checked_mul(2).and_then(|n| u64::try_from(n).ok()).ok_or(
            MetalCommitError::ShapeOverflow("ordinary spectrum ring stride"),
        )?;
        let bytes_per_ring = ring_stride
            .checked_mul(crate::runtime::FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64)
            .and_then(|n| n.checked_mul(size_of::<i32>() as u64))
            .ok_or(MetalCommitError::ShapeOverflow(
                "ordinary spectrum ring bytes",
            ))?;
        if !matches!(D, 64 | 128 | 256 | 512)
            || buffer.length() == 0
            || !buffer.length().is_multiple_of(bytes_per_ring)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "ordinary spectrum extent".into(),
            ));
        }
        let rings = buffer.length() / bytes_per_ring;
        Ok(Self {
            buffer,
            layout: MatrixSpectrumLayout {
                prime_stride: rings * ring_stride,
                ring_stride,
                frequency_offset: D as u64,
            },
        })
    }
}

pub(crate) struct RecursiveMatrixPreparation {
    pub(crate) buffer: Arc<MatrixBuffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) prepare_time: Duration,
    layout: MatrixSpectrumLayout,
}

impl RecursiveMatrixPreparation {
    pub(crate) fn from_ordinary<const D: usize>(
        matrix: MatrixPreparation,
    ) -> Result<Self, MetalCommitError> {
        let layout = RecursiveMatrixView::from_ordinary_buffer::<D>(&matrix.buffer)?.layout;
        Ok(Self {
            buffer: matrix.buffer,
            bytes: matrix.bytes,
            cache_hit: matrix.cache_hit,
            prepare_time: matrix.prepare_time,
            layout,
        })
    }

    pub(crate) fn view(&self) -> RecursiveMatrixView<'_> {
        RecursiveMatrixView {
            buffer: &self.buffer,
            layout: self.layout,
        }
    }
}

const _: [(); 24] = [(); size_of::<MatrixSpectrumLayout>()];

const _: [(); 8] = [(); std::mem::align_of::<MatrixSpectrumLayout>()];
