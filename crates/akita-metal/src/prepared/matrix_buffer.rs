use super::*;
use std::ops::{Deref, Range};

/// Clones retain a no-copy source owner through every escaped matrix handle.
#[derive(Clone)]
pub(crate) struct MatrixBuffer {
    // Drop the Metal reference before its source allocation.
    buffer: Buffer,
    owner: Option<Arc<AkitaExpandedSetup<F>>>,
}

impl MatrixBuffer {
    pub(crate) fn from_setup(
        runtime: &MetalRuntime,
        owner: &Arc<AkitaExpandedSetup<F>>,
        range: Range<usize>,
    ) -> Result<Self, AkitaError> {
        let fields = owner
            .shared_matrix()
            .as_field_slice()
            .get(range)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("canonical matrix slice is outside its setup".into())
            })?;
        let shared = runtime
            .shared_slice_buffer(fields)
            .map_err(MetalCommitError::into_akita)?;
        Ok(Self {
            buffer: shared.buffer,
            owner: shared.zero_copy.then(|| owner.clone()),
        })
    }

    pub(crate) fn zero_copy(&self) -> bool {
        self.owner.is_some()
    }

    pub(crate) fn allocation_bytes(&self) -> usize {
        if self.zero_copy() {
            0
        } else {
            self.buffer.length() as usize
        }
    }
}

impl From<Buffer> for MatrixBuffer {
    fn from(buffer: Buffer) -> Self {
        // This route is for independently allocated Metal storage, including
        // transformed matrices and unprepared field copies; it borrows no host memory.
        Self {
            buffer,
            owner: None,
        }
    }
}

impl Deref for MatrixBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        &self.buffer
    }
}
