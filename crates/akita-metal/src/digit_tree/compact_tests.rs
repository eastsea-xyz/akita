use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};
use akita_prover::protocol::sumcheck::digit_range::{
    ClassIndexedDigitTreeCompactSecondRoundInput, ClassIndexedDigitTreeCompactSourceInput,
    ClassIndexedDigitTreeCompactStageInput, ClassIndexedDigitTreeRound,
};
use akita_prover::{
    AkitaProverSetup, ComputeBackendSetup, CpuBackend, DigitRangeProver,
    DirectDigitRangeProofBackend, OperationCtx,
};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain, GrindingPlan, GrindingRun,
    GrindingSite, ProverGrindingTranscript, SetupMatrixCapacity, SumcheckProtocol,
};
use jolt_field::{CanonicalEncoding, One, Zero};

// Deterministic full-width samples keep parity failures reproducible without a new dependency.
fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_field(state: &mut u64) -> F {
    let high = u128::from(next_random(state));
    let low = u128::from(next_random(state));
    F::from_u128_reduced((high << 64) | low)
}

fn pair_polynomial(left: &[F], right: &[F], mode: &ClassIndexedDigitTreeRound<F>) -> [F; 5] {
    let mut result = [F::zero(); 5];
    match mode {
        ClassIndexedDigitTreeRound::Product {
            arity,
            parent_weights,
        } => {
            for (parent, &weight) in parent_weights.iter().enumerate() {
                let mut product = [F::one(), F::zero(), F::zero(), F::zero(), F::zero()];
                for lane in 0..*arity {
                    let index = parent * arity + lane;
                    let slope = right[index] - left[index];
                    let mut next = [F::zero(); 5];
                    for degree in 0..=lane {
                        next[degree] += product[degree] * left[index];
                        next[degree + 1] += product[degree] * slope;
                    }
                    product = next;
                }
                for degree in 0..5 {
                    result[degree] += weight * product[degree];
                }
            }
        }
        ClassIndexedDigitTreeRound::RangeLeaf { coefficients } => {
            let mut power = [F::one(), F::zero(), F::zero(), F::zero(), F::zero()];
            for (degree, &coefficient) in coefficients.iter().enumerate() {
                for index in 0..5 {
                    result[index] += coefficient * power[index];
                }
                if degree + 1 < coefficients.len() {
                    let mut next = [F::zero(); 5];
                    for index in 0..=degree {
                        next[index] += power[index] * left[0];
                        next[index + 1] += power[index] * (right[0] - left[0]);
                    }
                    power = next;
                }
            }
        }
    }
    result
}

#[test]
fn compact_tree_lut_rounds_materialization_and_resident_bind_match_cpu() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        16,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let factory = metal
        .class_indexed_digit_tree_factory(&prepared, 16384, 16)
        .unwrap()
        .unwrap();
    let mut seed = 0x73b6_2a91_5fe4_084d;
    for class_count in [8usize, 16, 32] {
        for mode in [
            ClassIndexedDigitTreeRound::Product {
                arity: 2,
                parent_weights: vec![random_field(&mut seed), random_field(&mut seed)],
            },
            ClassIndexedDigitTreeRound::Product {
                arity: 4,
                parent_weights: vec![random_field(&mut seed), random_field(&mut seed)],
            },
            ClassIndexedDigitTreeRound::RangeLeaf {
                coefficients: (0..5).map(|_| random_field(&mut seed)).collect(),
            },
        ] {
            let lanes = match &mode {
                ClassIndexedDigitTreeRound::Product {
                    arity,
                    parent_weights,
                } => arity * parent_weights.len(),
                ClassIndexedDigitTreeRound::RangeLeaf { .. } => 1,
            };
            let nodes = (0..class_count * lanes)
                .map(|_| random_field(&mut seed))
                .collect::<Vec<_>>();
            for live_len in [0usize, 1, 2, 64, 123, 128] {
                let classes = (0..live_len)
                    .map(|_| (next_random(&mut seed) as usize) % class_count)
                    .collect::<Vec<_>>();
                let indices = classes
                    .chunks(2)
                    .map(|pair| (pair[0] * class_count + pair.get(1).copied().unwrap_or(0)) as u16)
                    .collect::<Vec<_>>();
                factory
                    .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
                        domain_len: 128,
                        live_len,
                        class_count,
                        ordered_pairs: &indices,
                    })
                    .unwrap();
                let pair_lut_count = class_count * class_count;
                let first_lut = (0..pair_lut_count)
                    .map(|pair| {
                        let left = pair / class_count;
                        let right = pair % class_count;
                        pair_polynomial(
                            &nodes[left * lanes..(left + 1) * lanes],
                            &nodes[right * lanes..(right + 1) * lanes],
                            &mode,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut session = factory
                    .start_compact(ClassIndexedDigitTreeCompactStageInput {
                        pair_coefficients: &first_lut,
                        padding: &nodes[..lanes],
                        round: mode.clone(),
                    })
                    .unwrap()
                    .unwrap();
                let mut dense = classes
                    .iter()
                    .flat_map(|&class| nodes[class * lanes..(class + 1) * lanes].iter().copied())
                    .collect::<Vec<_>>();
                for _ in live_len..128 {
                    dense.extend_from_slice(&nodes[..lanes]);
                }
                let mut domain = 128usize;
                let mut explicit = live_len;
                let mut round = 0;
                while domain > 1 {
                    let first_len = (domain / 2).min(8);
                    let first = (0..first_len)
                        .map(|_| random_field(&mut seed))
                        .collect::<Vec<_>>();
                    let second = (0..domain / (2 * first_len))
                        .map(|_| random_field(&mut seed))
                        .collect::<Vec<_>>();
                    let mut suffix = F::zero();
                    for pair in explicit.div_ceil(2)..domain / 2 {
                        suffix += first[pair % first.len()] * second[pair / first.len()];
                    }
                    let actual = session.round_coefficients(&first, &second, suffix).unwrap();
                    let mut expected = [F::zero(); 5];
                    for pair in 0..domain / 2 {
                        let row = pair_polynomial(
                            &dense[2 * pair * lanes..(2 * pair + 1) * lanes],
                            &dense[(2 * pair + 1) * lanes..(2 * pair + 2) * lanes],
                            &mode,
                        );
                        let weight = first[pair % first.len()] * second[pair / first.len()];
                        for degree in 0..5 {
                            expected[degree] += weight * row[degree];
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "class_count={class_count}, round={round}, lanes={lanes}"
                    );
                    let challenge = random_field(&mut seed);
                    let final_value = session.ingest_challenge(challenge).unwrap();
                    for pair in 0..domain / 2 {
                        for lane in 0..lanes {
                            let left = dense[2 * pair * lanes + lane];
                            let right = dense[(2 * pair + 1) * lanes + lane];
                            dense[pair * lanes + lane] = left + challenge * (right - left);
                        }
                    }
                    dense.truncate(dense.len() / 2);
                    domain /= 2;
                    explicit = explicit.div_ceil(2);
                    if round == 0 {
                        let folded = (0..pair_lut_count)
                            .flat_map(|pair| {
                                let left = pair / class_count;
                                let right = pair % class_count;
                                (0..lanes)
                                    .map(|lane| {
                                        let left = nodes[left * lanes + lane];
                                        left + challenge * (nodes[right * lanes + lane] - left)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let quartet = (class_count == 8).then(|| {
                            (0..pair_lut_count * pair_lut_count)
                                .map(|index| {
                                    let left = index / pair_lut_count;
                                    let right = index % pair_lut_count;
                                    pair_polynomial(
                                        &folded[left * lanes..(left + 1) * lanes],
                                        &folded[right * lanes..(right + 1) * lanes],
                                        &mode,
                                    )
                                })
                                .collect::<Vec<_>>()
                        });
                        session
                            .prepare_compact_second_round(
                                ClassIndexedDigitTreeCompactSecondRoundInput {
                                    folded_pairs: &folded,
                                    quartet_coefficients: quartet.as_deref(),
                                },
                            )
                            .unwrap();
                    }
                    if domain == 1 {
                        assert_eq!(final_value.unwrap(), dense);
                    } else {
                        assert!(final_value.is_none());
                    }
                    round += 1;
                }
            }
        }
    }
}

#[test]
fn compact_tree_rejects_bad_source_and_missing_second_round_lut() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        16,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let factory = metal
        .class_indexed_digit_tree_factory(&prepared, 16384, 16)
        .unwrap()
        .unwrap();
    for indices in [vec![0u16; 63], vec![64u16; 64]] {
        assert!(factory
            .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
                domain_len: 128,
                live_len: 128,
                class_count: 8,
                ordered_pairs: &indices,
            })
            .is_err());
    }
    let mut bad_padding = vec![0u16; 64];
    bad_padding[63] = 1;
    assert!(factory
        .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
            domain_len: 128,
            live_len: 127,
            class_count: 8,
            ordered_pairs: &bad_padding,
        })
        .is_err());
    let coefficients = vec![[F::zero(); 5]; 64];
    let padding = [F::zero()];
    let indices = vec![0u16; 64];
    factory
        .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
            domain_len: 128,
            live_len: 128,
            class_count: 8,
            ordered_pairs: &indices,
        })
        .unwrap();
    factory
        .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
            domain_len: 4,
            live_len: 4,
            class_count: 8,
            ordered_pairs: &indices[..2],
        })
        .unwrap();
    assert!(factory
        .start_compact(ClassIndexedDigitTreeCompactStageInput {
            pair_coefficients: &coefficients,
            padding: &padding,
            round: ClassIndexedDigitTreeRound::RangeLeaf {
                coefficients: vec![F::zero()]
            },
        })
        .unwrap()
        .is_none());
    factory
        .prepare_compact_source(ClassIndexedDigitTreeCompactSourceInput {
            domain_len: 128,
            live_len: 128,
            class_count: 8,
            ordered_pairs: &indices,
        })
        .unwrap();
    let mut session = factory
        .start_compact(ClassIndexedDigitTreeCompactStageInput {
            pair_coefficients: &coefficients,
            padding: &padding,
            round: ClassIndexedDigitTreeRound::RangeLeaf {
                coefficients: vec![F::zero()],
            },
        })
        .unwrap()
        .unwrap();
    let folded = vec![F::zero(); 64];
    assert!(session
        .prepare_compact_second_round(ClassIndexedDigitTreeCompactSecondRoundInput {
            folded_pairs: &folded,
            quartet_coefficients: None,
        })
        .is_err());
    session
        .round_coefficients(&[F::one()], &[F::one(); 64], F::zero())
        .unwrap();
    session.ingest_challenge(F::one()).unwrap();
    assert!(session
        .round_coefficients(&[F::one()], &[F::one(); 32], F::zero())
        .is_err());
    session
        .prepare_compact_second_round(ClassIndexedDigitTreeCompactSecondRoundInput {
            folded_pairs: &folded,
            quartet_coefficients: None,
        })
        .unwrap();
    assert_eq!(
        session
            .round_coefficients(&[F::one()], &[F::one(); 32], F::zero())
            .unwrap(),
        [F::zero(); 5]
    );
}

fn four_row_proof<B>(
    backend: &B,
    prepared: &B::PreparedSetup,
    setup: &AkitaProverSetup<F>,
) -> (
    akita_types::AkitaStage1Proof<F>,
    Vec<F>,
    akita_types::TranscriptNonceStream,
    F,
)
where
    B: ComputeBackendSetup<F> + DirectDigitRangeProofBackend<F, F>,
{
    let plan = DigitRangePlan::new(16).unwrap();
    let mut runs = Vec::new();
    for stage in 0..=plan.product_stage_arities().len() {
        for round in 0..2 {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage1,
                        level: 0,
                        stage: stage as u32,
                        round,
                    },
                    1,
                    128,
                )
                .unwrap(),
            );
        }
        if stage < plan.product_stage_arities().len() {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::Stage1InterstageBatch {
                        level: 0,
                        stage: stage as u32,
                    },
                    1,
                    128,
                )
                .unwrap(),
            );
        }
    }
    let grinding = GrindingPlan::new(runs, 128).unwrap();
    let mut base = AkitaTranscript::<F>::new(b"akita-metal/four-row-compact-cpu-fallback");
    let mut transcript = ProverGrindingTranscript::new(&mut base, &grinding).unwrap();
    let prover = DigitRangeProver::new(
        std::sync::Arc::<[i8]>::from([0, -2, 3]),
        plan,
        FlatBooleanDomain::new(3, 2).unwrap(),
        DigitRangeEqualityPoint::from_column_then_ring_challenges(&[F::one(); 2], 2, 0).unwrap(),
    )
    .unwrap();
    let ctx = OperationCtx::new(backend, prepared, setup.expanded.as_ref()).unwrap();
    let (proof, point) = prover
        .prove_with_backend::<F, _, B>(&ctx, &mut transcript, None, 0)
        .unwrap();
    let nonces = transcript.finish().unwrap();
    let next = base.challenge_scalar(b"compact-cpu-fallback-after-stage");
    (proof, point, nonces, next)
}

#[test]
fn compact_tree_four_row_domain_preserves_cpu_proof_and_nonce() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        16,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let metal_prepared = metal.prepare_setup(&setup).unwrap();
    assert_eq!(
        four_row_proof(&cpu, &cpu_prepared, &setup),
        four_row_proof(&metal, &metal_prepared, &setup)
    );
}
