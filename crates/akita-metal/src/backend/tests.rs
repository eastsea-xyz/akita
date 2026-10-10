use std::sync::Arc;

use akita_prover::{
    AkitaProverSetup, ComputeBackendSetup, CpuBackend, DigitRangeProver,
    DirectDigitRangeProofBackend, OperationCtx,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain, GrindingPlan, GrindingRun,
    GrindingSite, ProverGrindingTranscript, SetupMatrixCapacity, SumcheckProtocol,
};
use jolt_field::Ring;

use super::*;
use crate::MetalExecutionPolicy;

const COLUMN_VARS: usize = 6;
const RING_VARS: usize = 9;

/// Balanced digits inside the range the basis admits, spread deterministically.
fn digit_witness(basis: usize) -> Arc<[i8]> {
    let live_len = 1usize << (COLUMN_VARS + RING_VARS);
    let (low, high): (i32, i32) = match basis {
        4 => (-2, 1),
        8 => (-4, 3),
        _ => unreachable!("direct range parity test covers basis four and eight"),
    };
    let span = (high - low + 1) as usize;
    (0..live_len)
        .map(|index| {
            let mixed = (index.wrapping_mul(2_654_435_761)) >> 7;
            i8::try_from(low + (mixed % span) as i32).unwrap()
        })
        .collect()
}

fn direct_range_proof<B>(
    backend: &B,
    prepared: &B::PreparedSetup,
    expanded: &AkitaExpandedSetup<F>,
    basis: usize,
    digits: &Arc<[i8]>,
) -> (AkitaStage1Proof<F>, Vec<F>)
where
    B: ComputeBackendSetup<F> + DirectDigitRangeProofBackend<F, F>,
{
    let num_vars = COLUMN_VARS + RING_VARS;
    let domain = FlatBooleanDomain::new(digits.len(), num_vars).unwrap();
    let challenges = (0..num_vars)
        .map(|index| F::from_u64(7 * index as u64 + 3))
        .collect::<Vec<_>>();
    let point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &challenges,
        COLUMN_VARS,
        RING_VARS,
    )
    .unwrap();
    let prover = DigitRangeProver::new(
        digits.clone(),
        DigitRangePlan::new(basis).unwrap(),
        domain,
        point,
    )
    .unwrap();
    let ctx = OperationCtx::new(backend, prepared, expanded).unwrap();
    let runs = (0..num_vars)
        .map(|round| {
            GrindingRun::proof_of_work(
                GrindingSite::SumcheckRound {
                    protocol: SumcheckProtocol::Stage1,
                    level: 0,
                    stage: 0,
                    round: round as u32,
                },
                1,
                128,
            )
            .unwrap()
        })
        .collect();
    let plan = GrindingPlan::new(runs, 128).unwrap();
    let mut transcript = AkitaTranscript::<F>::new(b"akita-metal/direct-digit-range-parity");
    let mut transcript = ProverGrindingTranscript::new(&mut transcript, &plan).unwrap();
    prover
        .prove_with_backend::<F, _, B>(&ctx, &mut transcript, None, 0)
        .unwrap()
}

/// The Metal direct digit-range prover must emit exactly the CPU prover's
/// round messages and final evaluations; the first-round message comes from
/// a separate kernel and has drifted from the fold rounds before.
fn assert_direct_range_matches_cpu(basis: usize) {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: 512,
        },
    )
    .unwrap();
    let digits = digit_witness(basis);

    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let expected = direct_range_proof(&cpu, &cpu_prepared, setup.expanded.as_ref(), basis, &digits);

    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let metal_prepared = metal.prepare_setup(&setup).unwrap();
    metal.begin_opening_metrics().unwrap();
    let actual = direct_range_proof(
        &metal,
        &metal_prepared,
        setup.expanded.as_ref(),
        basis,
        &digits,
    );
    for (stage, (actual_stage, expected_stage)) in
        actual.0.stages.iter().zip(&expected.0.stages).enumerate()
    {
        for (round, (actual_poly, expected_poly)) in actual_stage
            .sumcheck_proof
            .round_polys
            .iter()
            .zip(&expected_stage.sumcheck_proof.round_polys)
            .enumerate()
        {
            assert_eq!(
                actual_poly, expected_poly,
                "Metal Stage-1 direct digit-range round message diverges from the CPU \
                 prover at basis {basis}, stage {stage}, round {round}"
            );
        }
    }
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1, expected.1);
    let metrics = metal.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
}

#[test]
fn direct_digit_range_matches_cpu_for_basis_four() {
    assert_direct_range_matches_cpu(4);
}

#[test]
fn direct_digit_range_matches_cpu_for_basis_eight() {
    assert_direct_range_matches_cpu(8);
}

#[test]
fn centered_witness_z_gpu_matches_balanced_cpu_bytes_and_bounds() {
    use akita_prover::compute::DigitRowsComputeBackend;
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    fn check<const D: usize>(backend: &MetalBackend, state: &mut u64) {
        for basis in 1u32..=8 {
            let maximum_planes = ((128 + basis) / basis) as usize;
            for planes in [0, 1, 32u32.div_ceil(basis) as usize + 1, maximum_planes] {
                for rows in [0, 1, 3] {
                    let centered = (0..rows * D)
                        .map(|index| {
                            *state ^= *state << 13;
                            *state ^= *state >> 7;
                            *state ^= *state << 17;
                            match index % 13 {
                                0 => i32::MIN,
                                1 => i32::MAX,
                                2 => 0,
                                3 => -(1i32 << (basis - 1)),
                                4 => (1i32 << (basis - 1)) - 1,
                                _ => *state as i32,
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut digits = vec![0i8; rows * planes * D];
                    for row in 0..rows {
                        for coefficient in 0..D {
                            let mut value = centered[row * D + coefficient] as i128;
                            for plane in 0..planes {
                                let base = 1i128 << basis;
                                let raw = value & (base - 1);
                                let digit = if raw >= base / 2 { raw - base } else { raw };
                                digits[(row * planes + plane) * D + coefficient] = digit as i8;
                                value = (value - digit) >> basis;
                            }
                        }
                    }
                    for width in [basis as u8, 8] {
                        let mut bytes = vec![0u8; (digits.len() * usize::from(width)).div_ceil(8)];
                        let mut negative_abs_max = 0;
                        let mut positive_max = 0;
                        for (index, &digit) in digits.iter().enumerate() {
                            negative_abs_max = negative_abs_max.max(if digit < 0 {
                                digit.unsigned_abs()
                            } else {
                                0
                            });
                            positive_max =
                                positive_max.max(if digit > 0 { digit as u8 } else { 0 });
                            let raw = (digit as u8) & (u8::MAX >> (8 - width));
                            for bit in 0..usize::from(width) {
                                let position = index * usize::from(width) + bit;
                                bytes[position / 8] |= ((raw >> bit) & 1) << (position % 8);
                            }
                        }
                        let got = <MetalBackend as DigitRowsComputeBackend<F>>::pack_centered_witness_z::<D>(
                            backend, &centered, planes, basis, width,
                        ).unwrap().expect("RequireMetal must execute admitted witness packing");
                        assert_eq!(
                            got.bytes, bytes,
                            "D={D}, basis={basis}, planes={planes}, rows={rows}, width={width}"
                        );
                        assert_eq!(got.digit_len, digits.len());
                        assert_eq!(
                            (got.negative_abs_max, got.positive_max),
                            (negative_abs_max, positive_max)
                        );
                    }
                }
            }
        }
    }
    let mut state = 0x5e31_6a17_42cb_805d;
    check::<64>(&backend, &mut state);
    check::<128>(&backend, &mut state);
    check::<256>(&backend, &mut state);
    check::<512>(&backend, &mut state);
    for (planes, basis, width, len) in [
        (1, 0, 3, 64),
        (1, 9, 8, 64),
        (1, 4, 3, 64),
        (46, 3, 3, 64),
        (1, 3, 3, 63),
    ] {
        assert!(
            <MetalBackend as DigitRowsComputeBackend<F>>::pack_centered_witness_z::<64>(
                &backend,
                &vec![0; len],
                planes,
                basis,
                width,
            )
            .is_err()
        );
    }
}
