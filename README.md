# vLLM on Kubernetes (AKS)

Running a quantized LLM on a GPU node pool in AKS, and autoscaling it without the replica count oscillating.

The repo is the working rig behind a short article series: the manifests, the ScaledObjects for each autoscaling attempt, the load generator used to produce the graphs, and the write-ups themselves.

**Stack:** Azure AKS 1.35.6 · `Standard_NV36ads_A10_v5` (1× A10, 24 GB) · `vllm/vllm-openai:latest` serving `Qwen/Qwen2.5-7B-Instruct-AWQ` · KEDA + Prometheus + Grafana.

---

## Articles

### [Choosing the Right GPU for Your Model — A Sizing Method, Not a Guess](write-up-gpu-sizing.md)

How to pick a GPU from a model's spec sheet instead of guessing. Weights are `parameters × bytes_per_parameter` (verify against the actual `.safetensors` sizes — the rule of thumb runs low on quantized models). What is left after weights and vLLM's overhead is KV cache, and `bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes` converts that into a token budget, then into concurrent requests. Runs the math both directions: VRAM → capacity, and a concurrency requirement → minimum VRAM. Closes with verifying the estimate against vLLM's startup log, plus when multiple GPUs change the math (tensor parallelism) and when they don't (more replicas).

### [Your vLLM Autoscaler Is Flapping Because You Picked the Wrong Signal — Not the Wrong Number](write-up-flapping.md)

Why queue-depth autoscaling flaps under steady load, through three attempts: an arbitrary threshold (over-scales, because HPA computes `ceil(total_waiting / threshold)`), a threshold calculated from measured capacity and an SLO (still flaps), and a stable design. The fix is the signal, not the number — `kv_cache_usage_perc` rises *before* requests queue, and querying it as a `sum` keeps a fixed point that does not move when a replica is added, where an `avg` reintroduces the oscillation. Also covers scale-down damping sized to real lead time, and why adding a second KEDA trigger does not give you AND logic — KEDA takes the max unless you use `scalingModifiers`.

---

## Contents

| Path | What it is |
|---|---|
| [deployment.yaml](deployment.yaml) | vLLM Deployment — GPU nodepool toleration/selector, `--gpu-memory-utilization 0.85`, `--max-num-seqs 32`, one GPU per pod |
| [service.yaml](service.yaml) | ClusterIP fronting the pods on port 8080 |
| [scaled_Objects/scaledobject-attempt1-guess.yaml](scaled_Objects/scaledobject-attempt1-guess.yaml) | Attempt 1 — arbitrary queue threshold (`5`). Example only, never run |
| [scaled_Objects/scaledobject-attempt2-calculated.yaml](scaled_Objects/scaledobject-attempt2-calculated.yaml) | Attempt 2 — calculated queue threshold (`7`), damping off. The flapping run |
| [scaled_Objects/scaledobject-attempt3-stabilized.yaml](scaled_Objects/scaledobject-attempt3-stabilized.yaml) | Attempt 3 — `sum(vllm:kv_cache_usage_perc)` at `0.80` + 120s scale-down window. The stable run |
| [scaled_Objects/scaledobject-scalingmodifiers.yaml](scaled_Objects/scaledobject-scalingmodifiers.yaml) | Both signals combined explicitly: `max(kv_cache / 0.80, num_waiting / 7)` |
| [load_test.py](load_test.py) | Async load generator — unique ~7,500-token prompts to defeat prefix caching, run in-cluster against the ClusterIP |
| [images/](images/) | Grafana captures for the flapping and stable runs |

---

## Running it

Assumes an AKS cluster with a GPU node pool labelled/tainted `sku=gpu`, plus KEDA and kube-prometheus-stack installed.

```bash
kubectl apply -f deployment.yaml -f service.yaml

# confirm the estimate against what vLLM actually measured
kubectl logs deploy/vllm-openai-gpu | grep -i "kv cache"

# pick one autoscaling attempt
kubectl apply -f scaled_Objects/scaledobject-attempt3-stabilized.yaml
```

The load generator must run **inside** the cluster — `kubectl port-forward` pins to a single pod, so scaled-up replicas would receive no traffic.

```bash
pip install -r requirements.txt   # httpx, requests, aiohttp
python load_test.py               # from a pod in the cluster
```

---

## Takeaways

1. Size the GPU from the spec sheet, then verify against vLLM's startup log — the estimate is a filter, the log is ground truth.
2. Autoscale on a leading signal (KV cache utilization), not a lagging one (queue depth).
3. Query it as a `sum`: a good signal's total is invariant to replica count.
4. Damp scale-down by at least your measured scale-up lead time.
