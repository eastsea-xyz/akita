use super::*;
use akita_types::{public_matrix_page_xof_input, sample_akita_setup_seed};
use jolt_field::CanonicalEncoding;

#[test]
fn gpu_prefix_matches_cpu_for_seeds_and_page_boundaries() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    for seed in [sample_akita_setup_seed(), AkitaSetupSeed::from([31; 32])] {
        for count in [0, 1, 4095, 4096, 4097, 8193] {
            let expected = derive_public_matrix_prefix::<F>(count, &seed);
            let output = backend.derive_public_matrix_prefix(count, &seed).unwrap();
            assert_eq!(output.matrix, expected);
            assert_eq!(output.metrics.output_bytes, count * size_of::<F>());
            assert_eq!(output.metrics.pages, count.div_ceil(4096));
            assert_eq!(output.metrics.gpu_field_elements, count);
            assert_eq!(output.metrics.command_buffers, usize::from(count != 0));
            assert!(!output.metrics.cpu_fallback);
            if count != 0 {
                assert!(output.metrics.command_wall_time > Duration::ZERO);
            } else {
                assert_eq!(output.metrics.peak_allocation_bytes, 0);
            }
        }
    }
}

#[test]
fn gpu_preserves_independent_pre_edit_shake_goldens() {
    // Frozen independent hashlib SHAKE oracle; the existing Rust offset-275
    // golden vector was used to check the oracle's encoding interpretation.
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    for (seed, expected) in [
        (
            sample_akita_setup_seed(),
            [
                279_843_191_702_189_603_212_009_806_111_888_907_400u128,
                205_051_486_485_098_597_290_295_153_543_494_326_905,
                309_143_558_275_187_587_283_102_360_883_761_802_597,
            ],
        ),
        (
            AkitaSetupSeed::from([31; 32]),
            [
                0x0d1d_9873_e4f6_b149_1cc9_ebf9_fb5e_1bbe,
                0x7de0_ddc2_31b1_08bd_d023_e213_4c9c_cfe8,
                0xab18_1201_996b_971d_8c75_d92b_6a81_217d,
            ],
        ),
    ] {
        let output = backend.derive_public_matrix_prefix(4097, &seed).unwrap();
        let fields = output.matrix.as_field_slice();
        assert_eq!(
            [0, 4095, 4096].map(|i| fields[i].to_u128_checked().unwrap()),
            expected
        );
    }
}

#[test]
fn gpu_handles_nongolden_pages_and_high_page_index_bytes() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let runtime = backend.runtime().unwrap();
    let seed = AkitaSetupSeed::from([31; 32]);
    let expected = derive_public_matrix_prefix::<F>(3 * 4096 + 31, &seed);
    let page = runtime
        .dispatch_setup_matrix_with_limit(&seed, 3, 31, SETUP_MATRIX_DRAW_LIMIT)
        .unwrap();
    assert_eq!(page.fields, expected.as_field_slice()[3 * 4096..]);
    // These inputs exercise both halves of the page index and the split at
    // byte 272 of the 277-byte absorption message without huge host prefixes.
    for (index, value) in [
        (16_777_221u64, 0xe774_7b8f_5f8d_b280_1061_93c4_ec11_b7d3u128),
        (4_294_967_301, 0x8a23_af7e_315f_46c7_05f9_0b60_13cb_d8b0),
        (
            9_223_372_036_854_775_813,
            0x82bb_073b_3ae6_0914_9f4f_b43b_888a_7d6b,
        ),
    ] {
        let page = runtime
            .dispatch_setup_matrix_with_limit(&seed, index, 1, SETUP_MATRIX_DRAW_LIMIT)
            .unwrap();
        assert_eq!(page.fields[0].to_u128_checked(), Some(value));
    }
    let across_carry = runtime
        .dispatch_setup_matrix_with_limit(&seed, u64::from(u32::MAX), 4097, SETUP_MATRIX_DRAW_LIMIT)
        .unwrap();
    assert_eq!(
        across_carry.fields[0].to_u128_checked(),
        Some(0x24b9_864b_a613_1ffb_87d7_37f6_7203_942e)
    );
    assert_eq!(
        across_carry.fields[4096].to_u128_checked(),
        Some(0xed78_0dfc_6273_744c_fce1_5eb4_4446_698d)
    );
}

#[test]
fn shared_page_encoder_keeps_fixed_fields_and_u64_page_bytes() {
    let seed = AkitaSetupSeed::from([31; 32]);
    let index = 0x0102_0304_0506_0708;
    let message = public_matrix_page_xof_input::<F>(&seed, index);
    assert_eq!(message.len(), 277);
    assert_eq!(&message[136..144], &4096u64.to_le_bytes());
    assert_eq!(&message[164..196], &[31; 32]);
    assert_eq!(
        &message[217..249],
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 0, 0, 88, 9,
        ]
    );
    assert_eq!(&message[269..277], &index.to_le_bytes());
}

#[test]
fn gpu_rejection_sampling_advances_the_scripted_stream_without_reduction() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let modulus = 0xffff_ffff_ffff_ffff_ffff_ffff_0000_5809;
    let candidates =
        [modulus, u128::MAX, 42, modulus - 1, 5].map(crate::field::Fp128Limbs::from_u128);
    let (fields, counts) = backend
        .runtime()
        .unwrap()
        .setup_matrix_sampling_oracle(&candidates, 3)
        .unwrap();
    assert_eq!(counts, [3, 5]);
    assert_eq!(
        fields
            .iter()
            .map(|value| value.to_u128_checked().unwrap())
            .collect::<Vec<_>>(),
        [42, modulus - 1, 5]
    );
}

#[test]
fn real_gpu_budget_failure_propagates_or_uses_explicit_preferred_cpu_fallback() {
    let seed = sample_akita_setup_seed();
    let required = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let error = required
        .derive_setup_matrix_with_limit(3, &seed, 0)
        .unwrap_err();
    assert!(
        matches!(error, MetalCommitError::UnsupportedShape(ref message)
        if message.contains("rejection budget exhausted"))
    );
    let preferred = MetalBackend::new(MetalExecutionPolicy::PreferMetal).unwrap();
    let output = preferred
        .derive_setup_matrix_with_limit(3, &seed, 0)
        .unwrap();
    assert_eq!(output.matrix, derive_public_matrix_prefix::<F>(3, &seed));
    assert!(output.metrics.cpu_fallback);
    assert!(output
        .metrics
        .fallback_reason
        .unwrap()
        .contains("rejection budget exhausted"));
}

#[test]
fn invalid_size_and_page_ranges_are_rejected_before_gpu_allocation() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let runtime = backend.runtime().unwrap();
    let seed = sample_akita_setup_seed();
    for (first, count) in [
        (0, 0),
        (0, SETUP_MATRIX_MAX_PAGES * 4096 + 1),
        (u64::MAX, 4097),
    ] {
        assert!(runtime
            .dispatch_setup_matrix_with_limit(&seed, first, count, SETUP_MATRIX_DRAW_LIMIT)
            .is_err());
    }
    assert!(matches!(
        backend.derive_public_matrix_prefix(usize::MAX, &seed),
        Err(MetalCommitError::ShapeOverflow(_))
    ));
}

#[test]
fn multiple_gpu_windows_preserve_the_flat_prefix_and_bound_device_memory() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let seed = AkitaSetupSeed::from([31; 32]);
    let count = SETUP_MATRIX_MAX_PAGES * SETUP_MATRIX_FIELDS_PER_PAGE + 1;
    let expected = derive_public_matrix_prefix::<F>(count, &seed);
    let output = backend.derive_public_matrix_prefix(count, &seed).unwrap();
    assert_eq!(output.matrix, expected);
    assert_eq!(output.metrics.command_buffers, 2);
    assert_eq!(output.metrics.gpu_field_elements, count);
    assert!(output.metrics.peak_allocation_bytes <= 16 * 1024 * 1024 + 256 * 8 + 277);
    assert!(!output.metrics.cpu_fallback);
}
