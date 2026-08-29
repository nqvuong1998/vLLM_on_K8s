# Standing Up a GPU Cluster on AKS for vLLM

This article is *Part of a series on running vLLM on AKS* and walks through creating an AKS cluster with a GPU node pool, deploying vLLM onto it, and wiring up Prometheus and Grafana for visibility.

Companion pieces: [Choosing the right GPU](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-gpu-sizing.md) [Why your autoscaler flaps](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-flapping.md).*

## Setup Summary

- **Cloud:** Azure
- **GPU node:** `Standard_NV36ads_A10_v5` (1× A10, 24 GB)
- **Image / model:** `vllm/vllm-openai:latest` serving `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Observability:** kube-prometheus-stack (Prometheus + Grafana), KEDA, NVIDIA DCGM exporter

All commands below are bash. The steps are ordered and each one depends on the previous.

---

## Dependency chain

The build order follows one chain: **model → VRAM requirement → GPU SKU → region availability → quota.**

## Step 0 — Prerequisites (one-time, survives resource group deletion)

**GPU quota.** Request through Portal → Quotas → Compute → <desired region> 
This article: Requested `Standard NVADSA10v5 Family vCPUs = 108` in `westus` (108 = 3 nodes × 36 vCPUs, matching the autoscaler's `max-count 3` set in step 3).

Quota is granted per-subscription and survives resource group deletion, so this step happens once, not on every rebuild.
 > A quota is Azure's per-subscription limit on how much of a resource (here, GPU vCPUs in a specific VM family) you're allowed to provision at once. New subscriptions start at 0 for GPU families since it's expensive and can be abused. 
 > You need it because without an approval, az aks nodepool add for a GPU will fail outright. The request goes through manual Azure approval, so it has to happen before you plan to build.

**Prerequisites **
- Existing Azure Subscription:

Local tooling:
- Helm 3+
- kubectl
- Bash

Bash Variables to set for use through the setup
```bash
RG=<resource-group-name>
CLUSTER=<cluster-name>
LOCATION=<preferred-location>
```

---

## Step 1 — Create Resource group

```bash
az group create -n $RG -l $LOCATION
```

## Step 2 — Create AKS cluster, on a CPU system pool

```bash
az aks create -g $RG -n $CLUSTER \
  --node-count 1 --node-vm-size Standard_D2s_v5 \
  --generate-ssh-keys
```

> The GPU does not go on this pool. Every AKS cluster requires a system node pool for cluster-critical pods (CoreDNS, metrics-server), and system pools cannot scale to zero — so a GPU placed here runs, and bills, 24/7 regardless of load. A `D2s_v5` CPU node covers the system pods cheaply; the GPU pool created in step 3 is where scale-to-zero actually happens.

## Step 3 — GPU node pool, tainted and scaled from zero

```bash
az aks nodepool add \
  -g $RG --cluster-name $CLUSTER \
  -n gpu \
  --node-vm-size Standard_NV36ads_A10_v5 \
  --node-count 0 --enable-cluster-autoscaler --min-count 0 --max-count 3 \
  --node-taints sku=gpu:NoSchedule --labels sku=gpu
```
What each flag does:

- **`--node-count 0` + `--enable-cluster-autoscaler`** — scale-from-zero. No node exists until a pod requires one, so there's no GPU spend until step 7. The cluster autoscaler evaluates a pending pod against the pool's declared taints/labels/VM size to decide whether it would fit, then provisions a node if so — which is why the taint and label must be set on the pool at creation, not discovered later from a running node.
- **`--min-count 0 --max-count 3`** — bounds the pool between 0 and 3 nodes. (Design decision)
- **`--node-taints sku=gpu:NoSchedule`** — blocks ordinary CPU pods from landing on the GPU node once it exists.
- **`--labels sku=gpu`** — the label the vLLM pod's `nodeSelector` targets in step 7.

> Taint, toleration, and nodeSelector do three separate jobs: the taint repels pods by default, a toleration permits a specific pod to ignore that taint, and a nodeSelector steers a pod toward a specific node. A toleration alone doesn't guarantee placement — it only lifts the block. The vLLM pod spec in step 7 carries both the toleration and the nodeSelector because both are required.

> The above scaling sets the --node-count and --min-count to 0; this may or may not be desirable for your use case. Keeping a node or 2 warm can help with latency, but there is cost associated with that. Choose whichever best fits your use case.

## Step 4 — Get kubeconfig for cluster communication

```bash
az aks get-credentials -g $RG -n $CLUSTER
kubectl get nodes          # expect only the system node — the GPU pool is still at 0
```

## Step 5 — NVIDIA device plugin

AKS does not install this by default. Without it, a GPU node never advertises `nvidia.com/gpu` as an allocatable resource, and any pod requesting `nvidia.com/gpu: "1"` stays `Pending` indefinitely with no error.

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
```

The upstream DaemonSet only tolerates the standard `nvidia.com/gpu` taint, not the custom `sku=gpu` taint set in step 3, so it won't schedule onto the GPU node without a patch:

```bash
kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-","value":{"operator":"Exists"}}]'
```

A `Pending` vLLM pod looks identical whether the device plugin is missing, mis-scheduled, or the node just hasn't scaled up yet. `kubectl describe node -l sku=gpu` and checking for `nvidia.com/gpu` under `Allocatable` distinguishes between the three.

## Step 6 — Observability core: Prometheus + Grafana + KEDA

NOTE: Installed before the GPU node scales up, so the stack builds on the free CPU pool. Values file: `observability/kps-values.yaml` — 6-hour retention, an 8 Gi PV on `managed-csi`, Grafana on `ClusterIP`, Alertmanager disabled.

No StorageClass to create beforehand — AKS ships a built-in `managed-csi` class (provisioner `disk.csi.azure.com`, `WaitForFirstConsumer` binding), which the Prometheus PVC above uses directly.

Add the chart repos first (one-time per workstation):

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
```

```bash
helm upgrade --install kps prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f observability/kps-values.yaml --timeout 10m

helm upgrade --install keda kedacore/keda -n keda --create-namespace --timeout 5m
```

Verify:

```bash
kubectl get pods -n monitoring          # prometheus, grafana, kube-state-metrics Running
kubectl get pods -n keda                # keda-operator + metrics-apiserver Running
kubectl api-resources | grep scaledobject   # confirms KEDA's CRDs landed
```

`prometheus-node-exporter` is configured to tolerate `sku=gpu` (`operator: Exists`) in the values file, so it lands on the GPU node automatically once it scales up — no separate install step needed for CPU/memory/disk metrics from that node.

Grafana's admin password is `admin`, set in the values file. Acceptable for a cluster torn down daily; not for anything long-lived.

## Step 7 — Deploy vLLM

```bash
kubectl apply -f deployment.yaml
kubectl get pods -w
```

Expected sequence: `Pending` → cluster autoscaler provisions an A10 node (~3–5 min) → `ContainerCreating` → image pull (~1 min, 8.8 GB) → model weights load → `1/1 Running`.

The pod spec (`[deployment.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/deployment.yaml)`) is where the taint/toleration/nodeSelector from step 3 get consumed:

```yaml
spec:
  tolerations:
    - key: sku
      operator: Equal
      value: gpu
      effect: NoSchedule   # matches --node-taints sku=gpu:NoSchedule on the GPU nodepool
  nodeSelector:
    sku: gpu               # matches --labels sku=gpu on the GPU nodepool
  containers:
    - name: vllm-gpu
      image: vllm/vllm-openai:latest
      args:
        - --model
        - Qwen/Qwen2.5-7B-Instruct-AWQ
        - --quantization
        - awq
        - --gpu-memory-utilization
        - "0.85"
        - --max-num-seqs
        - "32"
      resources:
        limits:
          nvidia.com/gpu: "1"    # ensures only one pod per GPU node
```

Two settings worth explaining:

- **`--gpu-memory-utilization 0.85`, not the vLLM default of 0.92.** At 0.92, vLLM tried to claim 21.82 GiB of the 24 GB card but only ~21.37 GiB was free, and engine init failed. The model fits — this is a headroom setting, not a capacity limit. Full VRAM accounting is covered in [GPU sizing](write-up-gpu-sizing.md).
- **`nvidia.com/gpu: "1"`** in `resources.limits` is what makes one-pod-per-node a scheduling constraint rather than a convention: Kubernetes tracks the node's GPU as consumed once this pod is placed, so a second replica can't land on the same node and the autoscaler brings up a new one instead.

## Step 8 — Service + smoke test

```bash
kubectl apply -f [service.yaml](https://github.com/JDoornink/vLLM_on_K8s/blob/main/service.yaml)
kubectl get endpoints vllm-openai-gpu        # must show pod IP:8080, confirming the selector matched

kubectl port-forward svc/vllm-openai-gpu 8080:8080 &
curl -s localhost:8080/v1/models | jq        # should list id "vllm-openai-gpu"
```

The Service is `ClusterIP` — reachable only via `port-forward`, no public IP. Switch to `type: LoadBalancer` only if the endpoint needs to be reached from outside the cluster (e.g., load-testing from a separate machine).

## Step 9 — GPU metrics: DCGM exporter

vLLM's `/metrics` endpoint reports request/queue stats but nothing about the GPU itself — no utilization, VRAM, temperature, or power. NVIDIA's DCGM exporter is a DaemonSet that reads the GPU directly and exposes it to Prometheus. It requires the GPU node to already be up (step 7) and, like the device plugin, must tolerate the `sku=gpu` taint to schedule there.

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update gpu-helm-charts
helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  -n monitoring -f observability/dcgm-values.yaml --timeout 5m

kubectl rollout status ds/dcgm-exporter -n monitoring --timeout=120s
```

Two failure modes, both fixed in `observability/dcgm-values.yaml`:

- **OOMKilled at a 256Mi memory limit (exit 137).** DCGM's field collection needs more headroom than that. Fixed with a `1Gi` limit.
- **`scrapeTimeout` must be ≤ `interval`.** The chart defaults `scrapeTimeout` to 25s; with `interval: 15s`, that combination makes the generated `ServiceMonitor` invalid, and the Prometheus operator drops the target with no visible error — the `ServiceMonitor` and `Service` objects both exist, Prometheus is healthy, but the target never appears. Fixed by setting `serviceMonitor.scrapeTimeout: 10s`.

## Step 10 — Wire vLLM into Prometheus

DCGM ships its own `ServiceMonitor`, discovered automatically because `kps-values.yaml` sets `serviceMonitorSelectorNilUsesHelmValues: false` (Prometheus picks up every monitor object in the cluster, not just ones with a specific release label). vLLM needs a `PodMonitor` instead, since it's scraped directly on the pod's metrics port:

```bash
kubectl apply -f observability/vllm-podmonitor.yaml
```

Verify both targets are `up`, not just present:

```bash
kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/targets \
  | jq -r '.data.activeTargets[] | select(.labels.job|test("vllm|dcgm";"i")) | "\(.health)  \(.labels.job)"'
# expect:  up  default/vllm-openai-gpu   AND   up  dcgm-exporter

curl -s 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL'      # GPU utilization series
curl -s 'http://localhost:9090/api/v1/query?query=vllm:num_requests_running' # vLLM's own series
```

## Step 11 — Access Grafana

```bash
kubectl port-forward -n monitoring svc/kps-grafana 3000:80 &
# browse http://localhost:3000  (admin / admin)
```

GPU-level and application-level metrics now land in the same Prometheus, on the same time axis — the data the capacity-planning math in [GPU sizing](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-gpu-sizing.md) and the autoscaler experiments in [the flapping article](https://github.com/JDoornink/vLLM_on_K8s/blob/main/write-up-flapping.md) are built from.

---

## Verification checklist

```bash
# device plugin landed on the GPU node and is Running
kubectl get pods -n kube-system -o wide | grep -i nvidia

# GPU is now advertised as an allocatable resource
kubectl describe node -l sku=gpu | grep -A8 Allocatable      # expect: nvidia.com/gpu: 1

# vLLM is actually serving
kubectl logs -f -l app=vllm-openai-gpu                       # look for "Application startup complete"
kubectl get pods -o wide                                     # confirm one pod per GPU node
```

If any of these fail, work backward through steps 5 → 3 rather than re-running the deployment. A `Pending` vLLM pod is almost always caused upstream of vLLM itself.

---

## Teardown

```bash
az group delete -n $RG --yes --no-wait
```

Deletes the cluster, both node pools, and the auto-created node resource group (`MC_*`) that holds the managed disks, including the Prometheus PV. The GPU quota grant from step 0 is untouched and persists for the next rebuild.

To keep the cluster but stop GPU spend without a full teardown:

```bash
kubectl scale deploy vllm-openai-gpu --replicas=0    # the GPU pool's autoscaler drains the node 1→0
```

---

## Gotchas

- **The GPU pool must be a separate user pool, not the default/system pool.** System pools can't scale to zero.
- **The NVIDIA device plugin needs an explicit toleration for a custom taint.** It only tolerates the standard `nvidia.com/gpu` taint by default.
- **`--gpu-memory-utilization 0.85`, not the 0.92 default**, on a 24 GB A10 — a headroom setting, not a capacity limit. See [GPU sizing](write-up-gpu-sizing.md).
- **DCGM's `scrapeTimeout` must be ≤ its scrape `interval`**, or the Prometheus operator drops the target with no visible error.
- **A crash-looping vLLM pod leaks GPU memory into the retry.** Delete the pod so VRAM releases before the next attempt.

## Summary

1. **Follow the dependency chain** — model → VRAM → GPU SKU → region → quota — and request quota ahead of time, since it starts at zero and requires manual approval.
2. **Keep the GPU pool separate from the system pool** so scale-to-zero is possible.
3. **Taint the GPU pool, and pair it with both a toleration and a nodeSelector on the pod.** A toleration alone only permits placement; it doesn't steer it.
4. **Install the NVIDIA device plugin manually** — AKS doesn't, and it needs the custom taint tolerated before it schedules.
5. **Tear down with the resource group, not the cluster alone** — it removes the managed disks and leaves quota untouched.

This gets the environment running. The remaining questions — how large a GPU the model actually needs, and what signal the autoscaler should watch — are covered in the companion articles [GPU sizing](write-up-gpu-sizing.md) and [the flapping article](write-up-flapping.md).
