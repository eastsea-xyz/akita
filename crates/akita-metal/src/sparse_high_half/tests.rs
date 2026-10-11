use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_prover::backend::RingSwitchRelationView;
use akita_prover::compute::{CpuBackend, RingSwitchRelationKernel, SparseHighHalfBatch};
use jolt_field::{CanonicalEncoding, Ring, Zero};

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}
fn ring<const D: usize>(state: &mut u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|index| match index % 17 {
        0 => F::zero(),
        1 => F::from_i64(-1),
        _ => F::from_u128_reduced((u128::from(next(state)) << 64) | u128::from(next(state))),
    }))
}
fn independent_high_half<const D: usize>(
    challenges: &[SparseChallenge],
    batch: SparseHighHalfBatch<'_, F, D>,
) -> CyclotomicRing<F, D> {
    let mut coefficients = [F::zero(); D];
    for (index, challenge) in challenges.iter().enumerate() {
        let source = index
            .checked_mul(batch.ring_stride)
            .and_then(|offset| offset.checked_add(batch.first_ring))
            .and_then(|index| batch.rings.get(index));
        let Some(source) = source else {
            continue;
        };
        for (&position, &weight) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            for (coefficient, destination) in coefficients.iter_mut().enumerate() {
                if coefficient < position as usize {
                    *destination += source.coefficients()[D + coefficient - position as usize]
                        * F::from_i64(i64::from(weight));
                }
            }
        }
    }
    let mut result = CyclotomicRing::from_coefficients(coefficients);
    if let Some(subtract) = batch.subtract {
        result -= *subtract;
    }
    result
}
fn parity<const D: usize>() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut state = 0x71fe_60a3_9cd5_284bu64;
    for claims in [1, 2] {
        for blocks in [1, 3, 7] {
            for a_rows in [1, 3] {
                let challenge_count = claims * blocks;
                let challenges = (0..challenge_count)
                    .map(|index| SparseChallenge {
                        positions: vec![
                            0,
                            (D - 1) as u32,
                            (next(&mut state) as usize % D) as u32,
                            (D - 1) as u32,
                            1,
                        ]
                        .into(),
                        coeffs: vec![-128, 127, if index % 2 == 0 { 2 } else { -2 }, -1, 0].into(),
                    })
                    .collect::<Vec<_>>();
                let folded = (0..challenge_count)
                    .map(|_| ring::<D>(&mut state))
                    .collect::<Vec<_>>();
                let recomposed = (0..challenge_count * a_rows)
                    .map(|_| ring::<D>(&mut state))
                    .collect::<Vec<_>>();
                let subtract = (0..a_rows)
                    .map(|_| ring::<D>(&mut state))
                    .collect::<Vec<_>>();
                let mut batches = vec![SparseHighHalfBatch {
                    rings: &folded,
                    first_ring: 0,
                    ring_stride: 1,
                    subtract: None,
                }];
                for (row, subtract_row) in subtract.iter().enumerate() {
                    batches.push(SparseHighHalfBatch {
                        rings: &recomposed,
                        first_ring: row,
                        ring_stride: a_rows,
                        subtract: Some(subtract_row),
                    });
                }
                batches.push(batches[1]);
                batches.push(SparseHighHalfBatch {
                    rings: &recomposed[..recomposed.len() - 1],
                    first_ring: a_rows - 1,
                    ring_stride: a_rows,
                    subtract: None,
                });
                let expected = batches
                    .iter()
                    .map(|&batch| independent_high_half(&challenges, batch))
                    .collect::<Vec<_>>();
                let cpu = <CpuBackend as RingSwitchRelationKernel<
                    RingSwitchRelationView<'_, D>,
                    F,
                    D,
                >>::sparse_high_half_batch(
                    &CpuBackend::DEFAULT, &challenges, &batches
                )
                .unwrap();
                assert_eq!(cpu, expected);
                backend.begin_opening_metrics().unwrap();
                let actual =
                    <MetalBackend as RingSwitchRelationKernel<
                        RingSwitchRelationView<'_, D>,
                        F,
                        D,
                    >>::sparse_high_half_batch(&backend, &challenges, &batches)
                    .unwrap();
                assert_eq!(
                    actual, expected,
                    "D={D}, claims={claims}, blocks={blocks}, A={a_rows}"
                );
                let metrics = backend.last_opening_metrics().unwrap().unwrap();
                assert!(
                    metrics.command_wall_time > std::time::Duration::ZERO,
                    "RequireMetal must submit this high-half batch to the GPU"
                );
                assert_eq!(metrics.cpu_fallback_calls, 0);
            }
        }
    }
    let invalid = SparseChallenge {
        positions: vec![D as u32].into(),
        coeffs: vec![1].into(),
    };
    assert!(<CpuBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(&CpuBackend::DEFAULT, &[invalid], &[]).is_err());
}
#[test]
fn sparse_high_half_random_strided_cpu_metal_parity_all_dimensions_requires_submission() {
    parity::<64>();
    parity::<128>();
    parity::<256>();
    parity::<512>();
}

fn assert_edge_batch<const D: usize>(
    backend: &MetalBackend,
    challenges: &[SparseChallenge],
    batches: &[SparseHighHalfBatch<'_, F, D>],
) {
    let expected = batches
        .iter()
        .map(|&batch| independent_high_half(challenges, batch))
        .collect::<Vec<_>>();
    assert_eq!(<CpuBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(
        &CpuBackend::DEFAULT, challenges, batches,
    ).unwrap(), expected);
    backend.begin_opening_metrics().unwrap();
    assert_eq!(<MetalBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(
        backend, challenges, batches,
    ).unwrap(), expected);
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    if batches.is_empty() {
        assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
    } else {
        assert!(
            metrics.command_wall_time > std::time::Duration::ZERO,
            "nonempty edge batch must submit GPU work"
        );
    }
    assert_eq!(metrics.cpu_fallback_calls, 0);
}

fn edges<const D: usize>() {
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut state = 0x89cb_140a_5d6e_237fu64;
    let sources = [ring::<D>(&mut state), ring::<D>(&mut state)];
    let subtract = ring::<D>(&mut state);
    let challenges = (0..3)
        .map(|_| SparseChallenge {
            positions: vec![1, (D - 1) as u32, (D - 1) as u32].into(),
            coeffs: vec![-128, 127, -1].into(),
        })
        .collect::<Vec<_>>();
    assert_edge_batch::<D>(
        &backend,
        &[],
        &[SparseHighHalfBatch {
            rings: &[],
            first_ring: 0,
            ring_stride: 1,
            subtract: Some(&subtract),
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[SparseHighHalfBatch {
            rings: &[],
            first_ring: 0,
            ring_stride: 1,
            subtract: None,
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[SparseHighHalfBatch {
            rings: &sources,
            first_ring: 1,
            ring_stride: 0,
            subtract: Some(&subtract),
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[SparseHighHalfBatch {
            rings: &sources,
            first_ring: sources.len(),
            ring_stride: 1,
            subtract: Some(&subtract),
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[SparseHighHalfBatch {
            rings: &sources,
            first_ring: usize::MAX,
            ring_stride: usize::MAX,
            subtract: None,
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[SparseHighHalfBatch {
            rings: &sources,
            first_ring: 1,
            ring_stride: usize::MAX,
            subtract: None,
        }],
    );
    assert_edge_batch::<D>(
        &backend,
        &challenges,
        &[
            SparseHighHalfBatch {
                rings: &sources[..1],
                first_ring: 0,
                ring_stride: 1,
                subtract: None,
            },
            SparseHighHalfBatch {
                rings: &sources,
                first_ring: 0,
                ring_stride: 1,
                subtract: None,
            },
        ],
    );
    assert_edge_batch::<D>(&backend, &challenges, &[]);
    let batch = [SparseHighHalfBatch {
        rings: &sources,
        first_ring: 0,
        ring_stride: 1,
        subtract: None,
    }];
    for invalid in [
        SparseChallenge {
            positions: vec![D as u32].into(),
            coeffs: vec![1].into(),
        },
        SparseChallenge {
            positions: vec![0, 1].into(),
            coeffs: vec![1].into(),
        },
    ] {
        assert!(<CpuBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(
            &CpuBackend::DEFAULT, std::slice::from_ref(&invalid), &batch,
        ).is_err());
        backend.begin_opening_metrics().unwrap();
        assert!(<MetalBackend as RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>>::sparse_high_half_batch(
            &backend, std::slice::from_ref(&invalid), &batch,
        ).is_err());
        let metrics = backend.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.command_wall_time, std::time::Duration::ZERO);
        assert_eq!(metrics.allocation_bytes, 0);
        assert_eq!(metrics.cpu_fallback_calls, 0);
    }
}

#[test]
fn sparse_high_half_empty_partial_and_overflow_views_match_cpu_before_submission_guards() {
    edges::<64>();
    edges::<128>();
    edges::<256>();
    edges::<512>();
}

#[repr(C, align(16384))]
struct AlignedSparseFields([F; 1024]);

fn aligned_canonical_upload<const D: usize>() {
    use crate::field::Fp128Limbs;
    use crate::runtime::sparse_high_half::{SparseHighHalfRow, SparseHighHalfTerm};
    use std::mem::{align_of, size_of, size_of_val};
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut state = 0x763a_14bf_9dc8_205eu64;
    let (actual, expected) = {
        let owner = Box::new(AlignedSparseFields(std::array::from_fn(|index| {
            match index % 13 {
                0 => F::zero(),
                1 => F::from_i64(-1),
                2 => F::from_u128_reduced(1u128 << 127),
                _ => F::from_u128_reduced(
                    (u128::from(next(&mut state)) << 64) | u128::from(next(&mut state)),
                ),
            }
        })));
        assert_eq!(size_of::<F>(), size_of::<Fp128Limbs>());
        assert_eq!(size_of::<CyclotomicRing<F, D>>(), D * size_of::<F>());
        assert_eq!(align_of::<CyclotomicRing<F, D>>(), align_of::<F>());
        assert_eq!(owner.0.as_ptr().addr() % 16384, 0);
        assert_eq!(size_of_val(&owner.0), 16384);
        for value in &owner.0 {
            // SAFETY: the concrete canonical F byte layout is checked against
            // its device conversion; unaligned reads impose no stronger F alignment.
            let raw =
                unsafe { std::ptr::read_unaligned(std::ptr::from_ref(value).cast::<Fp128Limbs>()) };
            assert_eq!(raw, Fp128Limbs::from_field(*value));
        }
        // SAFETY: the page is aligned for F/rings, fully initialized, and D
        // divides its length. The ring is transparent over [F; D], without padding.
        // The borrowed view stays inside owner scope until synchronous readback.
        let rings = unsafe {
            std::slice::from_raw_parts(
                owner.0.as_ptr().cast::<CyclotomicRing<F, D>>(),
                owner.0.len() / D,
            )
        };
        let challenge_count = rings.len();
        let challenges = (0..challenge_count)
            .map(|index| SparseChallenge {
                positions: vec![0, (D - 1) as u32, 1].into(),
                coeffs: vec![1, -128, if index % 2 == 0 { 127 } else { -2 }].into(),
            })
            .collect::<Vec<_>>();
        let subtract = ring::<D>(&mut state);
        let batches = [
            SparseHighHalfBatch {
                rings,
                first_ring: 0,
                ring_stride: 1,
                subtract: None,
            },
            SparseHighHalfBatch {
                rings,
                first_ring: 1,
                ring_stride: 2,
                subtract: Some(&subtract),
            },
            SparseHighHalfBatch {
                rings: &rings[..rings.len() - 1],
                first_ring: 0,
                ring_stride: usize::MAX,
                subtract: None,
            },
        ];
        let expected = batches
            .iter()
            .map(|&batch| independent_high_half(&challenges, batch))
            .collect::<Vec<_>>();
        backend.begin_opening_metrics().unwrap();
        let actual = <MetalBackend as RingSwitchRelationKernel<
            RingSwitchRelationView<'_, D>,
            F,
            D,
        >>::sparse_high_half_batch(&backend, &challenges, &batches)
        .unwrap();
        assert_eq!(actual, expected);
        let metrics = backend.last_opening_metrics().unwrap().unwrap();
        assert!(metrics.command_wall_time > std::time::Duration::ZERO);
        assert_eq!(metrics.cpu_fallback_calls, 0);
        let terms = 2 * challenge_count;
        let non_source_bytes = terms * size_of::<SparseHighHalfTerm>()
            + batches.len() * size_of::<SparseHighHalfRow>()
            + 2 * batches.len() * D * size_of::<Fp128Limbs>();
        assert_eq!(
            metrics.allocation_bytes, non_source_bytes,
            "aligned reused canonical source must borrow its page, without a source allocation"
        );
        (actual, expected)
    };
    // Owner storage has been dropped. Results must own completed canonical readback.
    assert_eq!(actual, expected);
}

#[test]
fn sparse_high_half_aligned_canonical_sources_reuse_borrowed_storage_until_readback() {
    aligned_canonical_upload::<64>();
    aligned_canonical_upload::<128>();
    aligned_canonical_upload::<256>();
    aligned_canonical_upload::<512>();
}
