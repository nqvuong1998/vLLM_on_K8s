import asyncio
import aiohttp
import time
import signal
import sys

# Runs in-cluster (loadgen pod) against the ClusterIP — kubectl port-forward pins
# to a single pod for its lifetime, so scaled-up replicas would get zero traffic.
URL = "http://vllm-openai-gpu.default.svc.cluster.local:8080/v1/completions"

# ~7,500-token prompt so the KV cache actually fills to ~0.80 at demo concurrency.
# Tokenizers don't compress repetition — each repeat costs the same tokens.
# Calibrate REPEATS via usage.prompt_tokens (FLAPPING-RUNBOOK.md step 0f).
PROMPT_BLOCK = (
    "Kubernetes autoscaling of GPU-backed LLM inference workloads requires careful "
    "attention to signal selection, capacity planning, and scale-down damping across "
    "every replica in the serving fleet. "
)
REPEATS = 225

# vLLM V1 enables automatic prefix caching by default: identical prompts share ONE
# cached KV copy, so 32 concurrent requests barely move kv_cache_usage_perc (~6% observed
# instead of ~80%). A unique lead-in per request breaks the shared prefix from token 0,
# forcing every request to occupy its own cache blocks like distinct real-world traffic.
import uuid


def make_payload() -> dict:
    return {
        "model": "vllm-openai-gpu",  # the --served-model-name, not the HF id
        "prompt": f"[session {uuid.uuid4()}] " + PROMPT_BLOCK * REPEATS,
        # 150, not 500: decode time dominates service time at batch ~30 (each sequence
        # decodes at only ~5-15 tok/s when the batch is large). 500-token outputs pushed
        # avg completion past 100s and past client timeouts. Cache math is unaffected —
        # the 7,500-token prompt dominates KV usage, not the output.
        "max_tokens": 150,
    }


PAYLOAD = make_payload()  # kept for the step 0f smoke test (single request)
REQUEST_TIMEOUT_S = 300  # generous: a timed-out client makes vLLM cancel the request, polluting the run

# Steady load: (concurrency, hold_duration_seconds). Both option runs use the same load.
# One pod sustains ~0.46 req/s (measured: rate(e2e_request_latency_seconds_count[2m]) at
# saturation). 3 requests / 5s = 0.6 req/s = 1.3x one pod (overloads it, queue grows
# ~8/min) and 65% of two pods (they absorb it) — the band the flap demo needs.
# 720s ≈ 2 full flap cycles at ~3-4 min/cycle (cycle includes ~90s pod startup).
STEPS = [(3, 720)]
BURST_INTERVAL_S = 5  # gap between bursts within a step

# Signals the whole run to stop early (e.g. Ctrl+C)
shutdown_event = asyncio.Event()


async def send_request(session: aiohttp.ClientSession, step_label: str, req_id: int):
    """Send one request, bounded by a timeout, and report how it went."""
    start = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with session.post(URL, json=make_payload(), timeout=timeout) as resp:
            await resp.read()  # drain body so the connection can be reused
            elapsed = time.monotonic() - start
            print(f"[step={step_label}] req={req_id} status={resp.status} elapsed={elapsed:.2f}s")
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        print(f"[step={step_label}] req={req_id} TIMED OUT after {elapsed:.2f}s")
    except asyncio.CancelledError:
        # Happens if shutdown fires mid-request. Let it propagate so the
        # gather() below actually stops waiting on us instead of hanging.
        print(f"[step={step_label}] req={req_id} cancelled")
        raise
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"[step={step_label}] req={req_id} FAILED after {elapsed:.2f}s: {e}")


async def run_step(session: aiohttp.ClientSession, concurrency: int, duration_s: int, step_label: str):
    print(f"\n=== STEP {step_label}: concurrency={concurrency}, duration={duration_s}s ===")
    end_time = time.monotonic() + duration_s
    req_id = 0
    in_flight: list[asyncio.Task] = []

    while time.monotonic() < end_time and not shutdown_event.is_set():
        burst = [
            asyncio.create_task(send_request(session, step_label, req_id + i))
            for i in range(concurrency)
        ]
        req_id += concurrency
        in_flight.extend(burst)

        # Wait for either the burst interval to pass or a shutdown signal,
        # whichever comes first — this is what makes Ctrl+C responsive
        # instead of blocking on a plain time.sleep().
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=BURST_INTERVAL_S)
        except asyncio.TimeoutError:
            pass  # normal case: interval elapsed, shutdown wasn't requested

    # Let any requests still in flight for this step finish (or get
    # cancelled below if shutdown was requested) before moving to the next step.
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


async def run_all_steps():
    # force_close: a fresh connection per request, so kube-proxy load-balances every
    # request across replicas — keep-alive would pin the pool to pods that existed
    # when the connections were first opened.
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for concurrency, duration in STEPS:
            if shutdown_event.is_set():
                break
            await run_step(session, concurrency, duration, f"c{concurrency}")

    print("\nLoad test complete." if not shutdown_event.is_set() else "\nLoad test stopped early.")


def install_signal_handlers(loop: asyncio.AbstractEventLoop):
    """Set the shutdown event on SIGINT/SIGTERM instead of letting a raw
    KeyboardInterrupt tear through an in-flight gather() uncleanly."""

    def _trigger_shutdown():
        print("\nShutdown requested — finishing in-flight requests, then stopping...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger_shutdown)
        except NotImplementedError:
            # add_signal_handler isn't available on Windows's default loop.
            # Fall back to a no-op here; run_main()'s try/except KeyboardInterrupt
            # below covers Ctrl+C on Windows instead.
            pass


async def run_main():
    loop = asyncio.get_running_loop()
    install_signal_handlers(loop)
    await run_all_steps()


if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        # Windows fallback path: add_signal_handler wasn't available above,
        # so Ctrl+C surfaces here instead. Exit cleanly rather than dumping
        # a traceback.
        print("\nInterrupted. Exiting.")
        sys.exit(1)