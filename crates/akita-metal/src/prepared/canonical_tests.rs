use super::*;
use crate::runtime::{DigitRowsParams, FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL};
use crate::{profiling, MetalBackend, MetalExecutionPolicy};
use akita_algebra::CyclotomicRing;
use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
use akita_types::{AkitaSetupDescriptor, FlatMatrix, SetupMatrixCapacity};
use jolt_field::{CanonicalEncoding, Ring, Zero};
use metal::MTLStorageMode;

fn aligned_fixture() -> (Arc<AkitaExpandedSetup<F>>, std::ops::Range<usize>) {
    let alignment = 16 * 1024;
    let fields_per_page = alignment / size_of::<F>();
    let fields = (0..3 * fields_per_page)
        .map(|index| match index % 7 {
            0 => F::zero(),
            1 => F::from_i8(-1),
            2 => F::from_u128_reduced(1u128 << 127),
            _ => F::from_i64(index as i64 * 37 - 40000),
        })
        .collect::<Vec<_>>();
    let owner = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 20,
                max_num_batched_polys: 1,
                num_field_elements: fields.len(),
                setup_seed: akita_types::sample_akita_setup_seed(),
            },
            FlatMatrix::from_flat_data(fields),
        ),
    );
    let pointer = owner.shared_matrix().as_field_slice().as_ptr().addr();
    let offset = (alignment - pointer % alignment) % alignment;
    assert!(offset.is_multiple_of(size_of::<F>()));
    let start = offset / size_of::<F>();
    (owner, start..start + fields_per_page)
}

fn check_owned_gpu_matrix(
    owner: &Arc<AkitaExpandedSetup<F>>,
    range: std::ops::Range<usize>,
    zero_copy: bool,
) {
    let runtime = MetalRuntime::new().unwrap();
    let fields = &owner.shared_matrix().as_field_slice()[range.clone()];
    let matrix = MatrixBuffer::from_setup(&runtime, owner, range).unwrap();
    assert_eq!(matrix.zero_copy(), zero_copy);
    assert_eq!(
        matrix.allocation_bytes(),
        if zero_copy { 0 } else { size_of_val(fields) }
    );
    assert_eq!(matrix.storage_mode(), MTLStorageMode::Shared);
    assert_eq!(matrix.length() as usize, size_of_val(fields));
    assert_eq!(
        matrix.contents() == fields.as_ptr().cast_mut().cast(),
        zero_copy
    );
    // SAFETY: canonical shared buffer holds exactly the checked field slice.
    let actual = unsafe { std::slice::from_raw_parts(matrix.contents().cast::<F>(), fields.len()) };
    assert_eq!(actual, fields);
    let (digits, expected) = cpu_matrix_reference(fields);
    check_gpu_matrix(&runtime, &matrix, &digits, &expected);
}

fn cpu_matrix_reference(fields: &[F]) -> (Vec<[i8; 64]>, CyclotomicRing<F, 64>) {
    const D: usize = 64;
    assert!(fields.len().is_multiple_of(D));
    let columns = fields.len() / D;
    let digits = (0..columns)
        .map(|column| {
            std::array::from_fn(|coefficient| ((column * 3 + coefficient * 5) % 8) as i8 - 4)
        })
        .collect::<Vec<[i8; D]>>();
    let expected = fields.as_chunks::<D>().0.iter().zip(&digits).fold(
        CyclotomicRing::<F, D>::zero(),
        |sum, (field, digit)| {
            sum + CyclotomicRing::from_slice(field)
                * CyclotomicRing::from_coefficients(digit.map(F::from_i8))
        },
    );
    (digits, expected)
}

fn check_gpu_matrix(
    runtime: &MetalRuntime,
    matrix: &MatrixBuffer,
    digits: &[[i8; 64]],
    expected: &CyclotomicRing<F, 64>,
) {
    const D: usize = 64;
    let columns = digits.len();
    let outcome = runtime
        .dispatch_fp128_d64_digit_rows(
            matrix,
            &[digits],
            true,
            DigitRowsParams {
                num_vectors: 1,
                num_rows: 1,
                num_cols: columns as u64,
                ring_d: D as u64,
                output_coefficients: D as u64,
                columns_per_partial: FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64,
                column_partials: columns.div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL) as u64,
                retain_quotients: 1,
            },
        )
        .unwrap();
    let coefficients = outcome.coefficients[..D]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| F::from_device(value, index).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(CyclotomicRing::<F, D>::from_slice(&coefficients), *expected);
}

#[test]
fn canonical_matrix_aligned_and_unaligned_match_cpu_gpu_without_conversion() {
    let (owner, aligned) = aligned_fixture();
    check_owned_gpu_matrix(&owner, aligned.clone(), true);
    check_owned_gpu_matrix(&owner, aligned.start + 1..aligned.end + 1, false);
}

#[test]
fn canonical_matrix_no_copy_handles_retain_owner_after_drop_and_clone() {
    let runtime = MetalRuntime::new().unwrap();
    let (owner, range) = aligned_fixture();
    let expected = owner.shared_matrix().as_field_slice()[range.clone()].to_vec();
    let (digits, expected_gpu) = cpu_matrix_reference(&expected);
    let weak = Arc::downgrade(&owner);
    let buffer = MatrixBuffer::from_setup(&runtime, &owner, range).unwrap();
    let escaped = buffer.clone();
    assert!(escaped.zero_copy());
    drop(buffer);
    drop(owner);
    assert!(weak.upgrade().is_some());
    // SAFETY: escaped MatrixBuffer's Arc retains the original field allocation.
    let actual =
        unsafe { std::slice::from_raw_parts(escaped.contents().cast::<F>(), expected.len()) };
    assert_eq!(actual, expected);
    // Only the escaped clone retains the original aligned allocation here.
    check_gpu_matrix(&runtime, &escaped, &digits, &expected_gpu);
    drop(escaped);
    assert!(weak.upgrade().is_none());
}

#[test]
fn canonical_matrix_shared_copy_and_retained_runtime_sources_survive_owner_drop() {
    let runtime = MetalRuntime::new().unwrap();
    let (owner, range) = aligned_fixture();
    let expected = owner.shared_matrix().as_field_slice()[range.clone()].to_vec();
    let copied_range = range.start + 1..range.end + 1;
    let expected_copy = owner.shared_matrix().as_field_slice()[copied_range.clone()].to_vec();
    let (digits, expected_gpu) = cpu_matrix_reference(&expected);
    let (copy_digits, expected_copy_gpu) = cpu_matrix_reference(&expected_copy);
    let weak = Arc::downgrade(&owner);
    let retained = crate::runtime::RecursiveFoldSource::Field {
        buffer: MatrixBuffer::from_setup(&runtime, &owner, range).unwrap(),
    };
    let copied = MatrixBuffer::from_setup(&runtime, &owner, copied_range).unwrap();
    assert!(!copied.zero_copy());
    drop(owner);
    assert!(weak.upgrade().is_some());
    let crate::runtime::RecursiveFoldSource::Field { buffer } = &retained else {
        panic!("field source");
    };
    // SAFETY: the runtime source owns the no-copy field allocation; copied owns its shared storage.
    let actual =
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<F>(), expected.len()) };
    assert_eq!(actual, expected);
    check_gpu_matrix(&runtime, buffer, &digits, &expected_gpu);
    drop(retained);
    assert!(weak.upgrade().is_none());
    // SAFETY: independent shared copy remains initialized after the host owner is released.
    let actual_copy =
        unsafe { std::slice::from_raw_parts(copied.contents().cast::<F>(), expected_copy.len()) };
    assert_eq!(actual_copy, expected_copy);
    // The independent shared copy outlives every original host/setup handle.
    check_gpu_matrix(&runtime, &copied, &copy_digits, &expected_copy_gpu);
}

#[test]
fn canonical_matrix_growth_keeps_one_largest_prefix_and_checks_geometry() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4096,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let runtime = metal.runtime().unwrap();
    let first = prepared.matrix(runtime, 64, 1, 2).unwrap();
    let larger = prepared.matrix(runtime, 128, 3, 8).unwrap();
    assert!(!larger.cache_hit);
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    assert_eq!(
        prepared.matrix_cache_bytes().unwrap(),
        3 * 8 * 128 * size_of::<F>()
    );
    let small = prepared.matrix(runtime, 64, 2, 3).unwrap();
    assert!(small.cache_hit);
    assert!(Arc::ptr_eq(&larger.buffer, &small.buffer));
    // A cache hit must still reject unavailable matrix shapes.
    assert!(prepared.matrix(runtime, 128, 3, 32).is_err());
    assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    drop(prepared);
    drop(setup);
    assert_eq!(first.buffer.storage_mode(), MTLStorageMode::Shared);
    assert_eq!(larger.buffer.storage_mode(), MTLStorageMode::Shared);
    assert_eq!(small.buffer.storage_mode(), MTLStorageMode::Shared);
}

#[test]
fn canonical_matrix_preparation_submits_no_private_upload_blit() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 1024,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    profiling::set_enabled(true);
    let before = profiling::snapshot();
    let matrix = prepared
        .matrix(metal.runtime().unwrap(), 128, 1, 8)
        .unwrap();
    let after = profiling::snapshot();
    profiling::set_enabled(false);
    assert_eq!(
        after.submitted_command_buffers,
        before.submitted_command_buffers
    );
    assert_eq!(matrix.buffer.storage_mode(), MTLStorageMode::Shared);
}
