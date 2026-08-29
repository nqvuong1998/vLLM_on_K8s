# Choosing the Right GPU for Your Model — A Sizing Method, Not a Guess

## OK, you're a senior SRE, you've been hearing incessantly about AI models, but aren't quite sure how to determine the correct node size to host your model. - If so ... you're in the right place.

*Part of a series on running vLLM on AKS. Companion piece: [How to avoid flapping](https://dev.to/josef_doornink_930b2caf1c/your-vllm-autoscaler-is-flapping-because-you-picked-the-wrong-signal-not-the-wrong-number-24lf). GPU infrastructure setup — coming soon.*

This piece walks through estimating GPU memory requirements from both a model's parameter count or a concurrent requests reguirement. 

After reading this article you will have enough knowledge to pick a GPU family with confidence.

Disclaimer: this process is a rule-of-thumb filter, not a precise calculation — the last step covers how to get exact numbers once the model is actually running.


## Background: What actually consumes GPU memory

AI models live in GPU memory — VRAM — and engines such as vLLM provide novel techniques for managing that memory efficiently [[paper](https://arxiv.org/abs/2309.06180)], but the model isn't the only thing consuming it. 
Below is a short list of things that consume our precious VRAM:

1. **Model weights** — the parameters themselves. The big fixed cost: loaded once, never shrinks.
2. **KV cache** — working memory for in-flight requests. Every token of every active request holds its attention keys/values here. This is the one that determines *throughput*: more KV cache = more concurrent requests.
3. **Everything else** — activations (the temporary tensors of a forward pass) plus CUDA/framework overhead.  
You don't calculate these by hand; vLLM measures activations with a profiling pass at start-up and prints it for our consumption.

The sizing question is really: **after weights and overhead, how much is left for the KV cache — and is that enough for your traffic?**  

### OK, lets get started
---

## Step 1 — Choose a model

Guidance on *which* model to choose is outside the bounds of this article.  
What matters here: once you have a candidate, everything below can be read off its spec sheet — you can then run this method on every model on your shortlist and eliminate the ones that don't fit your requirements.

For demonstration purposes I will use Hugging Face's Qwen2.5-7B-Instruct-AWQ [model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ), [config card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/config.json)

---

## Step 2 — What to look for in the spec sheet

| What | Name | Value Qwen2.5-7B-Instruct-AWQ |
|---|---|---|
| Parameter count | [model card / Number of Paremeters](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) | ~7.6 B ("7B") |
| Quantization | `quantization_config` | AWQ, 4-bit |
| Layers | `num_hidden_layers` | 28 |
| KV heads | `num_key_value_heads` | 4 |
| Head dimension | `hidden_size / num_attention_heads` | 3584 / 28 = 128 |

The first two size the **weights**. The last three size the **KV cache per token**.  
That's the whole shopping list.

---

## Step 3 — Calculate the VRAM for the weights

Rule of thumb: `weights ≈ parameter_count × bytes_per_parameter` and many models have different precision offerings

| Precision | Bytes/param | 7.6 B params |
|---|---|---|
| fp16 / bf16 | 2 | ~15.2 GB |
| int8 | 1 | ~7.6 GB |
| AWQ / 4-bit | ~0.5 | ~3.8 GB |

Note on Weights and quantization: the above table compares the same model, just with different quantization:
**Quantization** is storing each weight in fewer bits than it was trained in. Models train in 16-bit float, so every parameter costs 2 bytes (16 bits); quantizing re-encodes them as 8-bit or 4-bit integers.

So the above table shows same model, same parameter count. Only the memory footprint for your model changes. 
Every GB you don't spend on weights is a GB left for KV cache, which allows serving more concurrent requests resulting in more happy customers.

Already this helps guide decisions:  
The fp16 model variant (~15 GB) plus workspace would nearly fill a 24 GB GPU before serving a single request. 
The AWQ variant (~5.6 GB) leaves the majority of VRAM free for the KV cache.  
Same model, same GPU — wildly different serving capacity.^^

> In truth, the AWQ actually consumes ~5.6 GB, not the advertise ~3.8 GB above. This can be seen by looking at the sum of the `.safetensors` file sizes on the repo's [Files and versions tab](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/tree/main) For this model that's exactly two files: [model-00001-of-00002.safetensors](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/model-00001-of-00002.safetensors) (~4.0 GB) and [model-00002-of-00002.safetensors](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/model-00002-of-00002.safetensors) (~1.6 GB). These are the weight tensors themselves — the files vLLM downloads and loads into VRAM at startup — so their combined size *is* the weights footprint (~5.6 GB). 

So now you know your weights footprint, - its time to look choose a GPU that fits your requirements.  
Below are links to some of the large cloud providers specification sheets.  
[GCP](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist), [AWS](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)

> **NOTE — reserve room for the engine:** vLLM pre-claims a fraction of total VRAM and is set using the `--gpu-memory-utilization` flag.  
> The vLLM engine fits weights + activations + KV cache inside that claim, leaving the remainder as headroom for CUDA overhead and fragmentation.  
> The default is 0.92; experiments showed that was too aggressive on our GPU and [we run 0.85](https://github.com/JDoornink/vLLM_on_K8s/blob/main/deployment.yaml).

So for the first pass, we are going to use Azures Standard_NV36ads_A10_v5 processor. (Note VRAM is listed under the Accelerators Tab in the [documents](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series?tabs=sizeaccelerators) — the "Memory (GiB)" column on the Basics tab is the VM's system RAM, not the GPU's.)
| Name | Accelerators | VRAM (GB) |
|---|---|---|
| Standard_NV36ads_A10_v5 | 1 | 24 |
---

## Step 4 — Determine your token budget. 

Now the centerpiece.  
On our A10 (24 GB): take 85% of it, subtract ~5.6 GB of weights and vLLM's measured activation/overhead reservation, and roughly **13.76 GiB** remains for the KV cache.

How many tokens fit in that? Each token in flight stores a key and a value vector in every layer:
```
bytes_per_token* = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
                = 2 × 28 × 4 × 128 × 2 (fp16)
                = 57,344 bytes ≈ 57 KB per token

[*Pope et al., 2022](https://arxiv.org/abs/2211.05102) 
Derivation assumes standard attention (MHA/MQA/GQA); sliding-window, MLA and hybrid SSM models cache differently and need a different formula.
```

Divide the remaining KV cache by the bytes per token:

```
Token budget  = 13.76 GiB / 57,344 bytes ≈ 257,584 tokens
```

And convert tokens into the unit you actually care about — concurrent requests (assuming ~1,000 tokens per request):

```
Concurrent_requests = 257,584 / 1,000 ≈ 258
```

One A10 can hold roughly **258 average requests in flight**. That single number is what connects GPU shopping to capacity planning — it's the same `Concurrent_requests` the [flapping article](https://dev.to/josef_doornink_930b2caf1c/your-vllm-autoscaler-is-flapping-because-you-picked-the-wrong-signal-not-the-wrong-number-24lf) builds its theoretical autoscaling threshold from.

---

## Step 5 — Pick the GPU from the requirements. (Which is what you will likely be doing anyway)

In reality the *requirement* is the fixed thing (ie 100 concurrent requests at peak) and the hardware is what you get to choose. 

So lets walk through how to solve for the GPU sizing from the given requirements 

```
KV bytes needed = concurrent_requests × avg_tokens_per_request × bytes_per_token
VRAM target     = (KV bytes + weights) / gpu_memory_utilization
```

Requirements/Assumptions — 100 concurrent requests at ~1,000 tokens each, same token size as above since we have already decided our model:

```
KV bytes needed = 100 × 1,000 × 57,344 bytes ≈ 5.7 GB
VRAM target = (5.7 + 5.6) / 0.85         ≈ 13.3 GB
```

13.3 GB is your shopping floor: any card below it can't hold this workload, and the fractional A10 sizes (4/8/12 GB) are eliminated on the spot. 
The A10's 24 GB clears it with room to spare — which is the useful answer, because "fits with headroom" is what lets you absorb a traffic spike without a second replica.

Now you have baseline requirements and can choose the model that fits those requirements BEFORE you've spent a dollar on hardware.

---

## Step 6 — Verify at boot: don't trust the estimate

OK, now the cluster and service is alive with the desired GPU (infrastructure setup is covered in a companion piece — coming soon).
The truth at startup is finally available becuase the truth comes from vLLM itself — at startup it profiles the hardware and prints exactly what it measured:.

```
INFO ... Available KV cache memory: 13.76 GiB
INFO ... GPU KV cache size: 257,584 tokens
```

`kubectl logs <vllm-pod> | grep -i "kv cache"` and compare against your Step 4 numbers. 
If they're close, your mental model of the card is correct.  
If they're way off, something in your assumptions is wrong (usually the quantization variant or the `gpu-memory-utilization` value) — better to find out now than after you've sized a node pool around it.

OK - there you go ... after a couple of tries you're officially an expert at determining the correct GPU to handle your inference model.
---


## Summary

1. **Model** — pick a candidate; everything below reads off its spec sheet, so you can run this on a whole shortlist.
2. **Spec sheet** — five values from the model card and `config.json`: parameter count, quantization, `num_hidden_layers`, `num_key_value_heads`, and `hidden_size / num_attention_heads`.
3. **Weights** — parameter count × bytes per parameter, then confirm against the real `.safetensors` file sizes (the rule of thumb runs low on quantized models).
4. **Token budget** — VRAM × `gpu-memory-utilization`, minus weights, ÷ bytes-per-token = tokens; ÷ tokens-per-request = `Concurrent_requests`. Note 0.85 is a safer starting point than vLLM's 0.92 default.
5. **Calculate GPU from Requests SLO** — start from the concurrency you actually need and solve for the VRAM target instead; anything below that number is off the shortlist.
6. **Verify** — the vLLM startup log is ground truth; grep it and reconcile against your Step 4 estimate.

Next in the series: that `Concurrent_requests` number is the foundation of a defensible autoscaling threshold — and why even a defensible threshold isn't enough: [Your vLLM Autoscaler Is Flapping Because You Picked the Wrong Signal](https://dev.to/josef_doornink_930b2caf1c/your-vllm-autoscaler-is-flapping-because-you-picked-the-wrong-signal-not-the-wrong-number-24lf).

---

## What about multiple GPUs?

Everything above sized *one* pod on *one* GPU. Fair question: what happens when the cluster has more than one? There are three scenarios here, and they're worth keeping distinct — because only one of them changes the math you just learned.

**Scenario 1 — many single-GPU nodes (what this series runs).** Each pod owns one GPU and holds a full copy of the model; traffic is load-balanced across replicas. This is *data parallelism*, and it's already a multi-GPU cluster — the [flapping article](https://dev.to/josef_doornink_930b2caf1c/your-vllm-autoscaler-is-flapping-because-you-picked-the-wrong-signal-not-the-wrong-number-24lf) scales exactly this fleet from 1 to 3 GPUs. Nothing in the sizing math changes: total capacity is simply `C × replicas`.

**Scenario 2 — multi-GPU nodes, still one GPU per pod.** Some VM sizes pack multiple cards (e.g. Azure's NV72ads_A10_v5 has 2× A10). Keep `nvidia.com/gpu: "1"` per pod and Kubernetes schedules two vLLM pods onto one node. The sizing math is *still* unchanged — each pod sees its own 24 GB card. This holds as long as [GPU time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) is disabled; with it enabled, two pods can share one physical card with no memory isolation and none of the numbers above apply.  

What does change is operational :

- **Scale-up gets faster** — a new replica can land on the already-running node in seconds, instead of waiting minutes for node provisioning.
- **Cost granularity gets coarser** — the node is the billing unit, so one busy pod next to an idle GPU slot still costs full price.
- **Blast radius grows** — losing one node now kills two replicas.

**Scenario 3 — multiple GPUs per *pod*: tensor parallelism. This is the one that changes the math.** When the model doesn't fit on any single card — Llama-70B at fp16 is ~140 GB of weights alone — vLLM can split every layer *across* GPUs with `--tensor-parallel-size N`, and the pod requests `nvidia.com/gpu: N`. Three consequences for sizing:

1. **Weights and KV cache are sharded** across the N GPUs — weights divide roughly evenly, and the `kv_heads` term in the bytes-per-token formula divides across GPUs too. (Note the constraint: with Qwen's 4 KV heads, tensor parallelism beyond 4 stops dividing cleanly.)
2. **Interconnect becomes a sizing axis.** Every forward pass exchanges activations between the GPUs, so NVLink vs. plain PCIe meaningfully changes throughput. VRAM stops being the only number you shop on.
3. **Autoscaling stakes go up.** One replica now costs N GPUs, so every flap cycle is N× more expensive — the signal-choice lessons from the flapping article matter *more* here, not less.

The clean rule of thumb: **more traffic → more replicas (Scenarios 1–2); bigger model → tensor parallelism (Scenario 3).** Reach for Scenario 3 only when weights plus a workable KV budget exceed the biggest single card you can buy — otherwise replicas are simpler, cheaper, and fail more gracefully.


## Gotchas

- **The 0.92 default failed on our A10.** vLLM tried to claim 21.82 GiB but only ~21.37 was actually free → engine init crashed. The model *fit* fine — this was a headroom knob, not a "need a bigger GPU" problem. `--gpu-memory-utilization 0.85` fixed it.
- **The weights rule of thumb underestimates quantized models.** `params × 0.5 bytes` says 3.8 GB; the real artifact is ~5.6 GB (fp16 embeddings + quantization scales). Check the actual file sizes on the repo.
- **Crash-looping pods leak VRAM.** If the vLLM pod crashes and restarts, the retry can fail on less-than-expected free VRAM. Delete the pod so the VRAM fully releases before the next attempt.

---
 