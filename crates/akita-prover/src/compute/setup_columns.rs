use akita_error::{checked, AkitaError};
use jolt_field::Field;
use std::ops::Range;

/// Scalar setup-ring evaluation followed by selected row-weight contraction.
///
/// Physical setup storage is row-major `[row][matrix column][ring coefficient]`.
/// Returned values are `[column offset within columns][weight batch]`.
#[derive(Clone, Debug)]
pub struct SetupColumnContractionPlan<'a, E: Field> {
    /// Number of rows in the checked physical setup prefix.
    pub matrix_rows: usize,
    /// Full matrix row width, before selecting a column range.
    pub matrix_columns: usize,
    /// Coefficients in each setup ring.
    pub ring_d: usize,
    /// Absolute columns to contract, retaining their original order.
    pub columns: Range<usize>,
    /// Selected physical row IDs and one scalar per batch. Repeated IDs add.
    pub row_weights: &'a [(usize, Vec<E>)],
    /// Exact `alpha^coefficient` evaluation weights in physical order.
    pub alpha_powers: &'a [E],
    /// Number of independent selected-row contractions per column.
    pub batch_count: usize,
}

impl<E: Field> SetupColumnContractionPlan<'_, E> {
    /// Validate every address and return the exact scalar output length.
    ///
    /// # Errors
    ///
    /// Rejects malformed weights, ranges, extents, or checked size overflow.
    pub fn validate(&self, setup_field_len: usize) -> Result<usize, AkitaError> {
        if self.ring_d == 0
            || self.batch_count == 0
            || self.alpha_powers.len() != self.ring_d
            || self.columns.start > self.columns.end
            || self.columns.end > self.matrix_columns
            || self
                .row_weights
                .iter()
                .any(|(row, weights)| *row >= self.matrix_rows || weights.len() != self.batch_count)
        {
            return Err(AkitaError::InvalidSetup(
                "setup column contraction shape is malformed".into(),
            ));
        }
        let fields = checked::product([self.matrix_rows, self.matrix_columns, self.ring_d])
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup column contraction matrix extent overflow".into())
            })?;
        if fields > setup_field_len {
            return Err(AkitaError::InvalidSize {
                expected: fields,
                actual: setup_field_len,
            });
        }
        checked::product([self.columns.len(), self.batch_count]).ok_or_else(|| {
            AkitaError::InvalidSetup("setup column contraction output extent overflow".into())
        })
    }
}
