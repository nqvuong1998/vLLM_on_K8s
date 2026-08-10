# Your vLLM Autoscaler Is Flapping Because You Picked the Wrong Signal — Not the Wrong Number

*Part of a series on running vLLM on AKS. Companion piece: [GPU sizing](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-gpu-sizing.md). Infrastructure setup — coming soon.*

So... you've picked the right GPU for your use case, provisioned the cluster and node pool, deployed your model behind vLLM, installed KEDA and picked the request-queue as the autoscaling parameter as per the common recommendation: ["Tune the request queue to obtain the preferred latency, and use batch size if you can't hit your preferred latency."](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)

Then you watch it in production:  
- A burst hits, a new pod spins up — and by the time it's ready, the burst is gone. The queue drained, so the autoscaler scales back down.  
- Minutes later the next spike repeats the cycle. The replica count oscillates instead of settling.  
- What gives??

That's **flapping** and it's costing you.

## What flapping is, and why you should care
Flapping refers to a resource rapidly switching between two states and in this case: a pod Up/Running or Down/Failed.

I can hear you say: "So what ... The requests still get served, right?"

True, but flapping costs you:
1. **Money** — you pay for GPU nodes that spin up and tear down without ever serving traffic.
2. **Latency** — the mechanism meant to *absorb* load spends its time flapping instead. Requests sit in the queue while pods churn.
3. **Trust** — pods and nodes appearing and vanishing "at random" is exactly the chaos you don't want when debugging under pressure.

This article shows how to avoid flapping by using the right *type* of signal and why request-queue is the wrong choice.

We demonstrate this by walking through three progressive designs for the autoscaler and its defining parameter(s), reasoning about the output, and finally coming to a conclusion about a signal that actually holds steady, and why.

The three progressive steps we will walk through are:

1. **Attempt 1 — arbitrary threshold** (flaps, and over-scales)
2. **Attempt 2 — calculated threshold** (still flaps)
3. **Attempt 3 — stabilized design** (stable)

## Setup

- **Cloud:** Azure AKS 1.35.6
- **GPU node:** Standard_NV36ads_A10_v5 (1× A10, 24 GB)
- **Image / model:** `vllm/vllm-openai:latest` / Qwen2.5-7B-Instruct-AWQ
- **Autoscaler:** KEDA ScaledObject, Prometheus trigger on `vllm:num_requests_waiting`
- **Visibility:** Prometheus + Grafana
- **Constraint:** `nvidia.com/gpu: "1"` → one pod per GPU node, so every replica needs a new node (this matters later)
---

## Attempt 1 — an arbitrary threshold (flaps, and over-scales)

Pick a low, round queue number that sounds reasonable (`5`) and move on — sound familiar?

This over-scales badly because the underlying Kubernetes HPA controller — which KEDA delegates the actual scaling decision to — computes `desiredReplicas = ceil(total_waiting / threshold)` ([Kubernetes HPA algorithm details](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details)). So a transient queue of 40 demands 8 replicas. Fine for a proof-of-concept, but in production each of those replicas cost you.

It's better to calculate the threshold from the hardware specs, not a guess.


So ... let's calculate a real one.  
Here is the scaledobject for attempt1 if you are curious. [scaledobject-attempt1-guess.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/scaled_Objects/scaledobject-attempt1-guess.yaml)

## Attempt 2 — a calculated threshold (still flaps)

Assumptions:
- Average tokens/request ≈ 1,000
- Average completion time = 10s (how long it takes vLLM to generate a response once processing starts — prefill + decode)
- Queue-wait budget `W_max` ≤ 5 seconds (SLO)

### To calculate the threshold, we first need to determine how much cache is available to our model
Read from the model's startup log (the engine profiles this at boot): this is VRAM left after weights + activation + overhead.
```
Result: Available KV cache memory: 13.76 GiB
```

Next we determine the bytes per token based on the model's specifications (all from the model's config.json):
```
bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
                = 2 × 28 × 4 × 128 × 2 (fp16) = 57,344

Calculate how many tokens are available given the cache size and the bytes required per token
257,584 ≈ 13.76 GiB / 57,344 bytes_per_token

Determine how many requests our GPU can handle at once given our assumption(s) (~1,000 tok/req)
Concurrent_requests = 257,584 tokens / 1,000 tokens/request ≈ 258
```

(Full VRAM → token-budget derivation is in the [GPU sizing](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-gpu-sizing.md) companion piece.)

Finally, calculate the **Threshold from an SLO** (Little's Law — how many requests may queue before wait breaches `W_max`):

```
# Calculate how many requests will be completed every second
Drain_rate = Concurrent_requests / avg_completion_time = 258 / 10s ≈ 25.8 req/s

# Calculate the threshold to increase the pod number by 1.
Threshold = W_max × Drain_rate = 5s × 25.8 ≈ 129
```

129 is a defensible number, derived from measured capacity and a stated SLO. Not a guess.

However, for demonstration purposes, the rest of this article runs on a deliberately shrunk rig:

> **Demo vs. production numbers:** `--max-num-seqs 32` caps concurrency at 32 so the queue is reachable at demo scale. Measuring with unique per-request prompts gave a real drain rate of `μ ≈ 0.46 req/s`; with a `W_max = 15s` budget, the same formula calibrates to `T = 15 × 0.46 ≈ 7`, versus `C ≈ 258` / `T ≈ 129` for the full-size pod. Don't get caught up on the smaller numbers — same method, same core point; it's just easier to demonstrate the flapping behavior and saves me money. [deployment.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/deployment.yaml)

Apply the ScaledObject ([scaledobject-attempt2-calculated.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/scaled_Objects/scaledobject-attempt2-calculated.yaml)):
```yaml
triggers:
  - type: prometheus
    metadata:
      serverAddress: http://kps-kube-prometheus-stack-prometheus.monitoring.svc:9090
      query: sum(vllm:num_requests_waiting)
      threshold: "7"
```

Apply the load ([load_test.py](https://github.com/JDoornink/vLLM_on_K8s/blob/main/load_test.py)):

![Attempt 2 — calculated threshold on queue depth, still flaps](https://raw.githubusercontent.com/JDoornink/vLLM_on_K8s/main/images/attempt2-calculated-threshold-with-lines.png)

Let's walk through what happened:

1. Requests start arriving, running count climbs to 32 (one pod's cap) — no queue yet, all requests are actually running.
2. Once running is saturated at 32, new arrivals start queuing instead. The queue grows.
3. Queue crosses 7 (the threshold) a little before 20:00 → KEDA scales to 2 replicas.
4. **While that second pod is still loading** (model weights + readiness, ~90s), the queue keeps growing — the new pod isn't serving yet, so it does nothing to relieve the backlog. By the time it's ready, the queue is way past 7 (peak ~50).
5. KEDA polls again, sees the queue still far over threshold, and scales to 3 pods.
6. Now with more capacity coming online, the queue starts draining. It falls below threshold, and KEDA scales back down (2, then 1) — the queue graph and replica graph fall together around 20:04–20:07.
7. The load never stopped: it's the same continuous arrival rate (0.6 req/s) throughout. The moment replicas dropped to 1 again, that single pod was immediately outmatched by the same steady arrivals, so the queue re-formed and crossed 7 again almost immediately (~20:08–20:10) — a second flap cycle.
8. Eventually the load generator's 12-minute run ends, arrivals stop, the queue drains for real, and replicas settle back to the floor of 1.

**Under steady load, the oscillation isn't coming from the traffic — it's coming from the shape of the signal.**  
The queue reads ~0 until the pod is full, then climbs; the autoscaler is just asking "is this threshold met or not?"  
The binary nature of the signal is what causes the flap.

> **Why you might not have noticed this in the wild:** Kubernetes' HPA defaults to a 300s scale-down stabilization window (0s scale-up), so out of the box the flap is slowed to a ~5-minute cycle — infrequent enough to miss at a glance, but you pay for it on every swing. ([Kubernetes HPA defaults](https://github.com/kubernetes/enhancements/blob/master/keps/sig-autoscaling/853-configurable-hpa-scale-velocity/README.md#default-values)) 
For this demo, `stabilizationWindowSeconds` was set to 0 in Attempt 2, to expose the flap within a short window instead of stretching it to 5+ minutes:
> ```yaml
> advanced:
>   horizontalPodAutoscalerConfig:
>     behavior:
>       scaleDown:
>         stabilizationWindowSeconds: 0
> ```

So even picking the "right" queue-depth number doesn't save you.
Is there a different metric we can use instead??

Now you're asking the right questions.

---

## Attempt 3 — the stabilized design (stable)

Two requirements to stabilize the system:
1. Scale on a signal that moves *before* trouble, and
2. Damp scale-down so a dip can't tear down capacity you just paid for.

First, let's look at some metrics and rank vLLM's metrics by *what they tell you and when*:

| Signal | Shape | Tells you |
|---|---|---|
| `num_requests_waiting` | 0 until full, then climbs | you're **already** overloaded (lagging) |
| `num_requests_running` | proportional, flat once at `max_num_seqs` | how busy, up to the cap |
| **`kv_cache_usage_perc`** | **smooth 0–1 (fraction)** | **how close to full — *before* the queue forms (leading)** |

KV-cache utilization is the right signal for vLLM because it rises **before** requests queue, and it *is* the binding resource the capacity math is about.
Scaling up at ~80% buys you lead time — you add capacity while the current pod still has headroom, INSTEAD OF AFTER it's drowning.

```yaml
triggers:
  - type: prometheus
    metadata:
      serverAddress: http://kps-kube-prometheus-stack-prometheus.monitoring.svc:9090
      query: sum(vllm:kv_cache_usage_perc)   # V1 engine name; older builds use gpu_cache_usage_perc
      threshold: "0.80"
```

**IMPORTANT: Use `sum`, not `avg` — this is the part that actually kills the flap.** `sum(kv_cache_usage_perc)` measures *total* demand across pods (in "pod-fulls"), which doesn't change when you add a replica — so `ceil(sum / 0.80)` has a **stable fixed point**. An `avg` falls as you add pods, drops back under the threshold, and reintroduces the exact oscillation you're trying to kill. That's the deeper rule behind this whole article: a good autoscaling signal is one whose **total is invariant to replica count**. `num_requests_waiting` fails it (collapses to 0 once you have enough capacity); `sum(kv_cache_usage_perc)` passes it.

OK, so how do we dampen the scale-down?

Pair the signal with **scale-down damping ≥ your measured lead time `L`**, so a momentary dip never tears down capacity you just paid minutes to bring up:

```yaml
advanced:
  horizontalPodAutoscalerConfig:
    behavior:
      scaleDown:
        stabilizationWindowSeconds: 120   # >= L (node + engine init + readiness)
```
Stable ScaledObject [scaledobject-attempt3-stabilized.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/scaled_Objects/scaledobject-attempt3-stabilized.yaml):

Apply the load again ([load_test.py](https://github.com/JDoornink/vLLM_on_K8s/blob/main/load_test.py)):

![Attempt 3 — kv_cache @ 80% + damping, replicas step up and hold, no oscillation](https://raw.githubusercontent.com/JDoornink/vLLM_on_K8s/main/images/attempt3-leading-signal-stable.png)

We now have a **stable** scale-up for processing.
It works without trying to find the exact right threshold from the request queue!
We don't even need to walk through the different points — the autoscaler handles it exactly like we want it to because we chose the right signal!

### Oh, big deal... "the docs already say to add a cache trigger"

They do — and that's the trap. Some ([e.g. this guide](https://dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead-1231)) suggest keeping the queue-depth trigger and *adding* a second cache trigger, assuming the two combine into "scale only if both agree."

**They don't, by default.** KEDA evaluates each trigger independently and takes the **max** of their desired replica counts — that's OR logic, not AND. So the lagging, bistable queue-depth trigger can still drive scale-up and scale-down on its own, and the flap comes right back. ([KEDA docs](https://keda.sh/docs/2.20/reference/faq/))

If you want the leading signal to govern scaling, **use it alone** — drop the queue-depth trigger. If you genuinely need both conditions, don't just list two triggers and hope: that silently keeps the OR/max default. AND-style logic is possible, but only if you explicitly use the opt-in [`scalingModifiers.formula`](https://keda.sh/docs/2.20/reference/scaledobject-spec/#scalingmodifiers) to combine the named triggers yourself. Adding a plain second trigger feels safer and quietly isn't.

## Summary

Flapping is a signal-choice bug wearing a threshold costume. In order of impact:

1. **Scale on a leading, proportional signal** (`kv_cache_usage_perc`), not a lagging saturation one (`num_requests_waiting`).
2. **Query it as a sum, not an average** — `sum(vllm:kv_cache_usage_perc)` measures *total* demand across pods, which doesn't change when you add a replica.
Additional Suggestions:
3. **Use scalingModifiers** if you need to combine metrics. Example here: [scaledobject-scalingmodifiers.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/scaled_Objects/scaledobject-scalingmodifiers.yaml).
4. **Size scale-down damping to your actual lead time.**

The final working scaledObject used for Attempt 3 can be seen [here](https://github.com/JDoornink/vLLM_on_K8s/blob/main/scaled_Objects/scaledobject-attempt3-stabilized.yaml) — trigger, threshold, and damping in one place.

You will likely still want to calculate your threshold, but — it's the last decision, not the first. 
Get the signal right and most of the flapping is gone before you tune anything.

---

## Alternatives

### Keep pods warm

You may not need the perfect signal. The cheapest lever against the timescale mismatch is **over-provisioning** — set `minReplicaCount > 1` so warm replicas absorb bursts without waiting minutes for a cold node.

What it fixes: the lead-time problem directly, and it makes any residual flapping **cheaper** (no cold starts) and **rarer** (you scale from a higher floor). What it doesn't fix: the marginal 2→3 decision still rides whatever signal you chose. And it costs money — idle GPUs are the expensive kind.

For a small deployment, **warm headroom + a reasonable threshold is often good enough** — the leading-signal work is where it pays off at scale or when you're cost-sensitive.

### More pods per node

If one-pod-per-GPU-node causes node churn, why not pack several pods per node? A couple of reasons this doesn't help:

1. Flapping is a control-loop stability problem — pack three pods per GPU and the replica count still oscillates, just faster.
2. Each pod gets a fraction of the 24 GB, so a fraction of the KV cache and a lower concurrent-request capacity (`C`).

---

## What changes at real scale

Everything above is one GPU — basically a teaching rig. Once you're running a real fleet, the strategy actually flips.

Instead of reacting fast to every queue blip, you want to **over-provision warm headroom and scale infrequently**. If possible, use **predictive** scaling — forecasting on daily/weekly patterns, not waiting on any live metric — for the traffic you can actually see coming, and let reactive scaling handle only the residual. Even that reactive layer should be the leading-signal kind from Attempt 3, not lagging queue-depth.

---

## What's next: KV-cache-aware routing

That's a topic for another article, but worth knowing it exists: at fleet scale, how you *route* requests across pods (e.g. KV-cache-aware routing) becomes another lever alongside scaling. Not a replacement for getting the scaling signal right — just the next layer once you have more than a few pods.

- [vLLM Production Stack — prefix-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/prefix-aware-routing.html)
- [Red Hat / llm-d — KV cache aware routing](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference)
