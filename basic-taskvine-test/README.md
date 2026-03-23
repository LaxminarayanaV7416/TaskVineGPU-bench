Perfect setup — 4 GPUs and all four proof points gives you a very complete experiment. Here's the full design:

---

## Recommended Multi-GPU Mechanism: DDP (DistributedDataParallel)

Use **DDP**, not DataParallel or FSDP, for these reasons:

- DDP launches **one process per GPU**, so TaskVine's process-based resource model is directly confronted — each rank is a real OS process that needs a real GPU
- The failure mode when given only 1 GPU is **immediate and unambiguous** — `dist.init_process_group` will fail or rank assignment will collide
- It's what real nanoGPT training uses in production, so it's scientifically credible
- FSDP is harder to set up and overkill for this proof; DataParallel is single-process so it obscures the scheduling argument

---

## The 4 Sub-Experiments

### Experiment A — Proof of Necessity (Task fails with 1 GPU)

**Design:** Submit a nanoGPT DDP task to TaskVine declaring only `gpus=1`. The task internally tries to launch 2 DDP ranks (GPT-2 medium is too large for 1 GPU at a meaningful batch size).

```
TaskVine Manager
    └── Task: "train nanoGPT with 2 DDP ranks"
            declared resources: gpus=1   ← intentionally wrong
            result: CRASH
```

The crash evidence you collect:
- `torch.distributed` backend init failure, OR
- CUDA OOM on rank 0 trying to hold the full model alone
- TaskVine reports task as failed, but its **reason** is wrong — it sees an exit code, not a GPU resource violation

**What this proves:** TaskVine has no way to know *why* the task failed. It can't distinguish "bad code" from "insufficient GPU resources declared". This is the resource declaration gap in action.

---

### Experiment B — Resource Declaration Gap

**Design:** Try to declare `gpus=2` in a TaskVine task and observe what the framework actually does with that declaration.

```python
# Manager side
t = vine.Task(command)
t.set_resources({"gpus": 2})   # Does TaskVine honor this?
```

You then verify with `nvidia-smi` on the worker side whether:
- TaskVine actually pins 2 specific GPU IDs to this task via `CUDA_VISIBLE_DEVICES`
- Or whether it just passes the declaration through and does no actual isolation
- Or whether `gpus > 1` is silently ignored/clamped to 1

Document the exact `CUDA_VISIBLE_DEVICES` value the task sees at runtime. This is your smoking gun for the scheduling gap.

---

### Experiment C — Worker GPU Inventory Tracking

**Design:** Submit 3 concurrent tasks, each correctly needing 2 GPUs, to a 4-GPU worker.

```
Worker: [GPU0, GPU1, GPU2, GPU3]

Expected TaskVine behavior:
    Task A → GPU0, GPU1    (running)
    Task B → GPU2, GPU3    (running)
    Task C → QUEUED        (no GPUs left)

Actual behavior to expose:
    Task A → dispatched
    Task B → dispatched
    Task C → also dispatched  ← inventory not tracked correctly
```

All 3 tasks then fight over 4 GPUs with 6 total GPU-slots demanded. Capture `nvidia-smi` showing all 3 training processes active simultaneously.

---

### Experiment D — Throughput Scaling (Justify Multi-GPU Scheduling)

**Design:** Run nanoGPT training as a *single TaskVine task* across 1, 2, and 4 GPUs and measure samples/second. This is your positive case — it proves multi-GPU scheduling is *worth building*.

```
Run 1: task declared gpus=1, DDP ranks=1  →  X samples/sec
Run 2: task declared gpus=2, DDP ranks=2  →  ~1.8X samples/sec
Run 4: task declared gpus=4, DDP ranks=4  →  ~3.5X samples/sec
```

Plot this curve. It becomes Figure 1 in your paper/report — the motivation for why TaskVine *needs* to support multi-GPU tasks correctly.

---

## How to Structure the TaskVine Manager

```
manager.py
    ├── Experiment A: submit 1-GPU task → expect failure, collect logs
    ├── Experiment B: submit gpus=2 task → inspect CUDA_VISIBLE_DEVICES
    ├── Experiment C: submit 3×(gpus=2) tasks → watch inventory behavior
    └── Experiment D: sweep gpus=1,2,4 → collect throughput metrics
```

Each task script should:
1. Print `os.environ.get('CUDA_VISIBLE_DEVICES')` immediately on startup — this is your ground truth for what TaskVine actually gave it
2. Use `torch.distributed` with `torchrun` or `mp.spawn` for DDP
3. Log samples/sec, epoch time, and GPU memory via `pynvml` every N steps
4. Write a JSON result file that the manager collects — so you have structured data, not just stdout

---

## The nanoGPT Model Size to Use

For a 4-GPU machine, use **GPT-2 Medium (345M params)** as your baseline:

| Config | 1 GPU behavior | 2 GPU behavior |
|---|---|---|
| GPT-2 Small (117M) | Fits fine — bad for proof of necessity | Marginal improvement |
| **GPT-2 Medium (345M)** | OOMs at large batch, or crawls | Scales well — good contrast |
| GPT-2 Large (774M) | Hard OOM | Fits comfortably |

GPT-2 Medium gives you the best of both worlds — it *can* fail on 1 GPU with a realistic batch size, and it *scales meaningfully* across 2 and 4 GPUs.

---

## Summary of What Each Experiment Proves

| Experiment | Proof Point | Failure Type |
|---|---|---|
| A | Task needs >1 GPU but TaskVine gives 1 | Hard crash |
| B | `gpus=N` declaration is not enforced | Scheduling/accounting bug |
| C | Worker doesn't track multi-GPU inventory | Over-scheduling bug |
| D | Multi-GPU gives real speedup | Positive motivation |



# -----------
# New ideas approach to prove that we need a scheduler in taskvine
# -----------

Great framing — you're essentially building a **stress test suite** that attacks the scheduler from multiple angles. Here are ideas organized by the *type of GPU need* they expose:

---

## Category 1 — Memory Hierarchy Failures

### 1A. Unified Memory Thrashing
**Model:** Any PyTorch model + `torch.cuda.memory.set_per_process_memory_fraction()`

CUDA Unified Memory allows the GPU to spill to CPU RAM silently. Submit two tasks that together exceed VRAM — instead of OOMing, they thrash the PCIe bus and slow to 10x normal speed. TaskVine sees both tasks as "running fine" while they're actually in a degraded state it has no visibility into.

**Proves:** TaskVine can't detect *soft* resource violations, only hard crashes.

---

### 1B. Memory Fragmentation Bomb
**Tool:** Repeated allocation/free cycles of varying sizes

```python
# Repeatedly allocate and free tensors of random sizes
# GPU allocator fragments → subsequent large contiguous 
# allocation fails even though free memory looks sufficient
for _ in range(1000):
    t = torch.randn(random_size, device='cuda')
    del t
# Now try to allocate one large contiguous block → OOM
```

**Proves:** VRAM accounting by bytes alone is insufficient — fragmentation matters and TaskVine has no model for it.

---

## Category 2 — Compute Contention (Not Just Memory)

### 2A. SM Saturation Without Memory Pressure
**Tool:** CUDA custom kernels via `triton` or `numba.cuda`

Write a kernel that pegs 100% of Streaming Multiprocessors but uses almost no memory. Two such tasks share a GPU — neither OOMs, but both slow down dramatically. `nvidia-smi` shows 100% GPU utilization split between two processes.

**Proves:** Memory is not the only resource — SM compute time needs scheduling too, and TaskVine has no SM-level isolation model.

---

### 2B. Tensor Core Starvation
**Tool:** Mixed-precision training (`torch.cuda.amp`)

Tensor Cores are a finite hardware resource on A100/V100. Two FP16 training tasks sharing a GPU will compete for Tensor Core access in ways that are invisible to any memory-based scheduler.

**Proves:** Modern GPUs have sub-resources (Tensor Cores, RT Cores, copy engines) that a robust scheduler needs to model.

---

## Category 3 — Multi-Process and Communication Failures

### 3A. NCCL Deadlock
**Tool:** nanoGPT or any DDP job + `torch.distributed` with NCCL backend

Submit two independent DDP jobs to the same worker. Each job initializes its own NCCL communicator. NCCL uses GPU-side lock primitives — two communicators on the same GPU can deadlock waiting for each other's collective operations to release the ring buffer.

**Proves:** GPU tasks aren't just isolated processes — they have *inter-process GPU state* (NCCL rings, IPC handles) that TaskVine's scheduler is completely unaware of.

---

### 3B. PCIe / NVLink Bandwidth Saturation
**Tool:** `torch.distributed` with explicit `device_to_device` transfers in a loop

```python
# Continuously stream large tensors GPU0 → GPU1
while training:
    send_tensor = model_shard.to('cuda:1')  # saturates NVLink
```

Submit a second task that also needs GPU-to-GPU bandwidth. Both tasks throttle. TaskVine scheduled them as independent — but they're contending on shared interconnect.

**Proves:** The GPU interconnect (NVLink/PCIe) is a schedulable resource, not just individual GPU cards.

---

## Category 4 — External Resource Dependencies

### 4A. CPU-GPU Pipeline Starvation
**Tool:** Heavy data augmentation pipeline (e.g. `albumentations` + DALI)

A GPU training task is only as fast as its CPU data loader. Submit enough tasks to saturate CPU cores — the GPU sits idle waiting for data. TaskVine sees "GPU underutilized" but can't see that CPU starvation is the root cause.

**Proves:** GPU tasks have *coupled* CPU + memory + I/O requirements that need co-scheduling, not just GPU slot allocation.

---

### 4B. Storage I/O Bottleneck
**Tool:** Large model checkpointing (`torch.save`) during training

Each task periodically saves a multi-GB checkpoint to shared NFS/disk. When multiple tasks checkpoint simultaneously, disk I/O saturates and GPU sits idle waiting for `torch.save` to complete.

**Proves:** GPU task scheduling must account for storage bandwidth as a co-resource.

---

### 4C. CUDA Context Overhead
**Tool:** Rapid task launch/teardown cycle

Submit 100 short GPU tasks (5–10 seconds each) in rapid succession. Each task pays ~500ms–1s for CUDA context initialization. This overhead is invisible to TaskVine — it schedules them as if launch is free, but a large fraction of wall time is wasted on context setup.

**Proves:** TaskVine needs a model for GPU task *startup cost*, not just runtime cost. Motivates GPU worker persistence / context reuse.

---

## Category 5 — Monitoring and Observability Gaps

### 5A. Silent Accuracy Degradation
**Tool:** ResNet or nanoGPT with checksum validation (from your earlier correctness experiment)

Two tasks share a GPU. Neither crashes, neither slows noticeably. But final model weights diverge from the single-task baseline due to non-deterministic kernel interleaving. TaskVine reports both tasks as **succeeded**.

**Proves:** TaskVine's success/failure signal is binary and insufficient — a task can "succeed" while producing scientifically invalid results.

---

### 5B. GPU Health Blindness
**Tool:** `pynvml` + deliberate thermal throttle trigger

Sustained compute at high utilization triggers thermal throttling on most GPUs — clock speed drops 20–40% automatically. TaskVine has no awareness of throttle state, so it keeps scheduling tasks at full speed assumptions.

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
```

**Proves:** A robust scheduler needs live GPU health telemetry, not just inventory counts.

---

### 5C. MIG Misconfiguration Blindness
**Tool:** NVIDIA MIG mode (if your A100 supports it)

Put the GPU in MIG mode creating unequal instances (e.g., one 3g.40gb and two 1g.10gb). Submit tasks that assume a full GPU. TaskVine has no MIG topology awareness — it may assign a 1g.10gb slice to a task that needs a full 40GB.

**Proves:** TaskVine needs GPU topology discovery, not just GPU count.

---

## Suggested Prioritization

| Priority | Experiment | Effort | Impact |
|---|---|---|---|
| ⭐⭐⭐ | 2A SM Saturation | Low | High — pure compute, no memory confusion |
| ⭐⭐⭐ | 3A NCCL Deadlock | Medium | Very high — dramatic, unambiguous failure |
| ⭐⭐⭐ | 5A Silent Accuracy | Low | High — builds on your existing ResNet code |
| ⭐⭐ | 4C CUDA Context Overhead | Low | Medium — good for short-task workloads argument |
| ⭐⭐ | 1A Unified Memory Thrashing | Medium | High — subtle, shows monitoring gap |
| ⭐⭐ | 4A CPU-GPU Pipeline | Medium | Good for co-scheduling argument |
| ⭐ | 5B Thermal Throttle | Low | Good observability argument |
| ⭐ | 5C MIG Blindness | High setup | Very high if hardware supports it |

---

The strongest **narrative arc** for your thesis would be:

```
Memory failures (ResNet OOM)
    → Compute contention (SM Saturation)
        → Multi-process failures (NCCL Deadlock)
            → Silent failures (Accuracy Degradation)
                → Observability gaps (Thermal / MIG)
```

Each layer adds a new dimension to why a robust GPU scheduler is necessary. Would you like to dive into implementation for any of these?
