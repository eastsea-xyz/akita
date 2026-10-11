use std::sync::Arc;

use akita_prover::protocol::sumcheck::digit_range::{
    ClassIndexedDigitTreeInput, ClassIndexedDigitTreeRound,
};
use akita_prover::{
    AkitaProverSetup, ComputeBackendSetup, CpuBackend, DigitRangeProver,
    DirectDigitRangeProofBackend, OperationCtx,
};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain, GrindingPlan, GrindingRun,
    GrindingSite, ProverGrindingTranscript, SetupMatrixCapacity, SumcheckProtocol,
    TranscriptNonceStream,
};
use jolt_field::{CanonicalEncoding, One, Zero};

use crate::field::F;
use crate::{MetalBackend, MetalExecutionPolicy};

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
fn resident_digit_tree_rounds_and_bind_match_padded_cpu_tables() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        16,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
        None,
    )
    .unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    let factory = metal
        .class_indexed_digit_tree_factory(&prepared, 16_384, 32)
        .unwrap()
        .unwrap();
    let mut random_state = 0xb64c_a790_375e_d819;
    let modes = [
        ClassIndexedDigitTreeRound::Product {
            arity: 2,
            parent_weights: vec![
                random_field(&mut random_state),
                random_field(&mut random_state),
            ],
        },
        ClassIndexedDigitTreeRound::Product {
            arity: 4,
            parent_weights: vec![
                random_field(&mut random_state),
                random_field(&mut random_state),
            ],
        },
        ClassIndexedDigitTreeRound::RangeLeaf {
            coefficients: (0..5).map(|_| random_field(&mut random_state)).collect(),
        },
    ];
    for mode in modes {
        let lanes = match &mode {
            ClassIndexedDigitTreeRound::Product {
                arity,
                parent_weights,
            } => arity * parent_weights.len(),
            ClassIndexedDigitTreeRound::RangeLeaf { .. } => 1,
        };
        for (initial_domain, explicit_lengths) in [
            (128_usize, vec![0_usize, 1, 3, 17, 65, 128]),
            (4_096, vec![1_025, 4_095, 4_096]),
        ] {
            for explicit_len in explicit_lengths {
                let padding = (0..lanes)
                    .map(|_| random_field(&mut random_state))
                    .collect::<Vec<_>>();
                let explicit = (0..explicit_len * lanes)
                    .map(|_| random_field(&mut random_state))
                    .collect::<Vec<_>>();
                let mut session = factory
                    .start(ClassIndexedDigitTreeInput {
                        domain_len: initial_domain,
                        explicit: &explicit,
                        padding: &padding,
                        round: mode.clone(),
                    })
                    .unwrap();
                let mut dense = explicit;
                for _ in explicit_len..initial_domain {
                    dense.extend_from_slice(&padding);
                }
                let mut domain = initial_domain;
                let mut remaining_explicit: usize = explicit_len;
                while domain > 1 {
                    let first_len = if domain == 2 { 1 } else { (domain / 2).min(8) };
                    let first = (0..first_len)
                        .map(|_| random_field(&mut random_state))
                        .collect::<Vec<_>>();
                    let second = (0..domain / (2 * first_len))
                        .map(|_| random_field(&mut random_state))
                        .collect::<Vec<_>>();
                    let pair_count = dense.len() / (2 * lanes);
                    let explicit_pairs = remaining_explicit.div_ceil(2).min(pair_count);
                    // The runtime computes the explicit prefix; the caller supplies the exact implicit suffix mass.
                    let mut suffix_weight = F::zero();
                    for pair in explicit_pairs..pair_count {
                        suffix_weight += first[pair % first.len()] * second[pair / first.len()];
                    }
                    let actual = session
                        .round_coefficients(&first, &second, suffix_weight)
                        .unwrap();
                    let mut expected = [F::zero(); 5];
                    for pair in 0..pair_count {
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
                    assert_eq!(actual, expected, "domain={domain}, lanes={lanes}");
                    let challenge = random_field(&mut random_state);
                    for pair in 0..pair_count {
                        for lane in 0..lanes {
                            let left = dense[2 * pair * lanes + lane];
                            let right = dense[(2 * pair + 1) * lanes + lane];
                            dense[pair * lanes + lane] = left + challenge * (right - left);
                        }
                    }
                    dense.truncate(dense.len() / 2);
                    let final_value = session.ingest_challenge(challenge).unwrap();
                    domain /= 2;
                    remaining_explicit = remaining_explicit.div_ceil(2);
                    if domain == 1 {
                        assert_eq!(final_value.unwrap(), dense);
                    }
                }
            }
        }
    }
}

fn tree_proof<B>(
    backend: &B,
    prepared: &B::PreparedSetup,
    setup: &AkitaProverSetup<F>,
    basis: usize,
    seed: u64,
) -> (
    akita_types::AkitaStage1Proof<F>,
    Vec<F>,
    TranscriptNonceStream,
    F,
)
where
    B: ComputeBackendSetup<F> + DirectDigitRangeProofBackend<F, F>,
{
    let num_vars = 16;
    let live_len = (1usize << num_vars) - 512;
    let mut random_state = seed;
    let digits: Arc<[i8]> = (0..live_len)
        .map(|_| (next_random(&mut random_state) % basis as u64) as i8 - (basis / 2) as i8)
        .collect();
    let tau = (0..num_vars)
        .map(|_| random_field(&mut random_state))
        .collect::<Vec<_>>();
    let point = DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau, 7, 9).unwrap();
    let digit_plan = DigitRangePlan::new(basis).unwrap();
    let mut runs = Vec::new();
    for stage in 0..=digit_plan.product_stage_arities().len() {
        for round in 0..num_vars {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage1,
                        level: 0,
                        stage: stage as u32,
                        round: round as u32,
                    },
                    1,
                    128,
                )
                .unwrap(),
            );
        }
        if stage < digit_plan.product_stage_arities().len() {
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
    let plan = GrindingPlan::new(runs, 128).unwrap();
    let mut base_transcript = AkitaTranscript::<F>::new(b"akita-metal/digit-tree-parity");
    let mut transcript = ProverGrindingTranscript::new(&mut base_transcript, &plan).unwrap();
    let prover = DigitRangeProver::new(
        digits,
        digit_plan,
        FlatBooleanDomain::new(live_len, num_vars).unwrap(),
        point,
    )
    .unwrap();
    let ctx = OperationCtx::new(backend, prepared, setup.expanded.as_ref()).unwrap();
    let (proof, point) = prover
        .prove_with_backend::<F, _, B>(&ctx, &mut transcript, None, 0)
        .unwrap();
    let nonces = transcript.finish().unwrap();
    let next_challenge = base_transcript.challenge_scalar(b"digit-tree-parity-after-stage");
    (proof, point, nonces, next_challenge)
}

#[test]
fn resident_digit_tree_preserves_cpu_tree_proof_and_child_claims() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
        None,
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let metal_prepared = metal.prepare_setup(&setup).unwrap();
    for basis in [16, 32, 64] {
        metal.begin_opening_metrics().unwrap();
        let expected = tree_proof(&cpu, &cpu_prepared, &setup, basis, 0xb64c_a790_375e_d819);
        let actual = tree_proof(
            &metal,
            &metal_prepared,
            &setup,
            basis,
            0xb64c_a790_375e_d819,
        );
        assert_eq!(actual, expected);
        assert!(
            metal
                .last_opening_metrics()
                .unwrap()
                .unwrap()
                .gpu_active_time
                .as_secs_f64()
                > 0.0
        );
    }
}
