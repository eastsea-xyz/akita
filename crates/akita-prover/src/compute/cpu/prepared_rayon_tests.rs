//! Cold-cache regressions isolated in owned child processes so a broken
//! OnceLock/Rayon initializer cannot hang the complete native test suite.

use super::*;
use jolt_field::Prime128OffsetA7F7;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_DOMAIN: &str = "AKITA_TEST_COLD_NTT_DOMAIN";
const CHILD_WORKERS: &str = "AKITA_TEST_COLD_NTT_WORKERS";
const CHILD_FIFO: &str = "AKITA_TEST_COLD_NTT_FIFO";
const CHILD_TIMEOUT: Duration = Duration::from_secs(10);

fn run_child(domain: &str, workers: usize, fifo: bool) {
    let full_name = concat!(module_path!(), "::cold_ntt_cache_child");
    let (_, test_name) = full_name
        .split_once("::")
        .expect("the test module is inside its crate");
    // Inherit TMPDIR and the parent's process group/guards. The child runs
    // only this CPU unit test, without acquiring another global proof lock.
    let mut child = Command::new(std::env::current_exe().expect("current native test executable"))
        .args(["--exact", test_name, "--ignored", "--test-threads=1"])
        .env(CHILD_DOMAIN, domain)
        .env(CHILD_WORKERS, workers.to_string())
        .env(CHILD_FIFO, if fifo { "1" } else { "0" })
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn the owned cold-cache regression child");
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the owned regression child") {
            break Some(status);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            // A deadlock terminates and reaps only the PID created above.
            // A process that exited between polls also gets reaped by wait.
            let _ = child.kill();
            child.wait().expect("reap the owned regression child");
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(
        status.is_some_and(|status| status.success()),
        "cold NTT initialization failed or timed out: domain={domain}, workers={workers}, fifo={fifo}",
    );
    let output = child
        .wait_with_output()
        .expect("collect the completed child test result");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
        "the child filter must execute exactly the cold-cache entry point",
    );
}

#[test]
fn same_cold_ntt_key_completes_on_one_fifo_worker() {
    for domain in ["negacyclic", "cyclic", "exact"] {
        run_child(domain, 1, true);
    }
}

#[test]
fn same_cold_ntt_key_builds_once_across_small_worker_pools() {
    for workers in [2, 4] {
        for domain in ["negacyclic", "cyclic", "exact"] {
            run_child(domain, workers, false);
        }
    }
}

#[expect(
    deprecated,
    reason = "one-worker FIFO order deterministically reproduces a stolen sibling re-entering OnceLock; scope_fifo does not change nested join queue order"
)]
fn cold_pool(workers: usize, fifo: bool) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("akita-cold-ntt-regression-{index}"));
    if fifo {
        builder = builder.breadth_first();
    }
    builder.build().expect("owned regression pool")
}

#[test]
#[ignore = "child entry point; both bounded parent tests invoke it explicitly"]
fn cold_ntt_cache_child() {
    const RING_D: usize = 512;
    const RINGS: usize = 32;
    let domain_name = std::env::var(CHILD_DOMAIN).expect("the bounded parent supplies a domain");
    let domain = [
        ("negacyclic", NttTransformDomain::Negacyclic),
        ("cyclic", NttTransformDomain::Cyclic),
        (
            "exact",
            NttTransformDomain::ExactNegacyclicI16 {
                width: RINGS,
                rhs_abs_bound: 1 << 15,
            },
        ),
    ]
    .into_iter()
    .find(|(name, _)| *name == domain_name)
    .map(|(_, domain)| domain)
    .expect("recognized cold-cache child domain");
    let workers = std::env::var(CHILD_WORKERS)
        .expect("the bounded parent supplies workers")
        .parse::<usize>()
        .expect("integer worker count");
    let fifo = std::env::var(CHILD_FIFO).expect("the bounded parent supplies queue order") == "1";
    let pool = cold_pool(workers, fifo);
    let setup = AkitaProverSetup::<Prime128OffsetA7F7>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: RINGS * RING_D,
        },
        None,
    )
    .expect("small fp128 setup without a guest");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("empty prepared setup");
    let key = NttCacheKey {
        ring_d: RING_D,
        num_ring_elements: RINGS,
        domain,
    };

    for generation in 1..=2 {
        let (left, right) = pool.install(|| {
            // With one FIFO worker, a parallel initializer's nested join
            // pops the older sibling before its own transform job. That
            // sibling blocks on the cell still owned by this stack frame.
            rayon::join(
                || CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, key),
                || CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, key),
            )
        });
        left.expect("left cold-cache request");
        right.expect("right cold-cache request");
        assert_eq!(prepared.ntt_slot_build_count(), generation);
        let metrics = prepared
            .shared_ntt_cache_metrics()
            .expect("completed cache metrics");
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics.first().expect("one completed entry").key, key);
        let bytes = prepared.shared_ntt_cache_bytes();
        assert!(bytes > 0);
        assert_eq!(
            bytes,
            prepared
                .planned_shared_ntt_cache_bytes([key])
                .expect("unchanged cache representation and policy"),
        );
        if generation == 1 {
            assert_eq!(
                prepared
                    .drop_built_ntt_slots()
                    .expect("retire completed cache"),
                bytes,
            );
            assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        }
    }
}
