
### Argument
TasVine needs an internal GPU aware monitoring system for better allocation of GPU resources and better utilization of GPU resources for better running collocated jobs on HTCondor or SLURM.

#### *Argument 1:*
##### Avoiding allocation of GPU Computes with alot of VRAM and SM being saturated by smaller tasks in HTCondor or SLURM that doesnt fully utilize the resources allocated. Intelligent GPU Resource Provisioning via Cooperative TaskVine–HTCondor Scheduling
* The Problem: Static and Uninformed GPU Allocation in HTC Environments
High-throughput computing frameworks like **HTCondor** and **SLURM** operate on a fundamentally static resource allocation model for GPU jobs. When a user submits a GPU job, they are expected to manually declare how much VRAM, how many GPUs, and how many SMs their job requires — upfront, before the job ever runs. In practice, most users either over-provision (requesting a full A100 when their workload only needs 8GB of VRAM) or under-provision (requesting too little and crashing at runtime). Neither outcome is acceptable at cluster scale.

More critically, **HTCondor has no native mechanism to introspect a job's actual GPU resource consumption** before scheduling it. It cannot query how much VRAM a PyTorch training script will allocate at peak, how aggressively it will saturate Streaming Multiprocessors, or whether it will use Tensor Cores at all. The result is a chronic pattern of **GPU fragmentation** — large, high-memory GPUs like A100s are monopolized by small tasks that only use a fraction of their capacity, while other jobs queue indefinitely waiting for a free GPU slot that is nominally occupied but computationally idle.

* The Proposed Solution: GPU Resource Profiler (Pre-Submission Profiling)
The core idea is to introduce a **GPU Resource Profiler** — a lightweight, automated profiling step that TaskVine executes *before* submitting any GPU task to HTCondor. Rather than asking the user to declare resource requirements manually, TaskVine runs the task in a **controlled, instrumented environment**, observes its actual GPU behavior, it actively intercepts and measures the GPU calls the task makes, building a ground-truth resource profile from inside the process, and uses that empirical data to construct a precise resource specification for HTCondor submission.

* Phase 1: Instrumented Dry Run and CUDA Call Interception when a user submits a GPU task to TaskVine, instead of immediately forwarding it to HTCondor, TaskVine first launches the task in a **profiling sandbox** on a local or staging GPU. During this dry run, TaskVine instruments the process at the CUDA API level by intercepting key calls:
- **`cudaMalloc` / `cudaMallocAsync`** — captures every GPU memory allocation event, its size, and the cumulative VRAM footprint over time, allowing TaskVine to identify **peak VRAM usage** rather than average usage
- **`cudaLaunchKernel`** — tracks the frequency and grid/block dimensions of kernel launches, from which SM occupancy can be estimated
- **`cublasGemmEx` / `cudnnConvolutionForward`** and similar library calls — identifies whether the task uses Tensor Cores (FP16/BF16 operations), which is relevant for matching the task to GPU architectures that have the right compute capabilities
- **`cudaMemcpy` / `cudaMemcpyAsync`** — measures host-to-device and device-to-device transfer volumes, which informs PCIe and NVLink bandwidth requirements
- **`ncclAllReduce` / `ncclBroadcast`** — detects whether the task uses collective communication, signaling that it requires multi-GPU topology awareness

This interception can be implemented via **LD_PRELOAD** — a standard Linux mechanism that injects a shared library into the process before execution, allowing TaskVine to wrap the real CUDA runtime calls with monitoring logic without modifying the user's code at all. The user's task runs completely unmodified; TaskVine simply observes it from the outside.
The dry run does not need to run to completion. TaskVine only needs to observe **one or two iterations** of the training loop — enough to capture the steady-state memory allocation pattern and compute behavior — before terminating the profiling run and moving to the next phase.

* Phase 2: Resource Profile Construction: From the instrumented dry run, TaskVine constructs a **GPU Resource Profile** for the task — a structured summary of its actual hardware needs:

* Phase 3: Cluster Inventory Query via `condor_status` With the resource profile in hand, TaskVine queries the HTCondor cluster's live inventory using `condor_status` to retrieve the full specification of every available GPU slot across all worker nodes (doesnt happen every time, we can cache this as well and perform this irregularly).
```bash
condor_status -compact -af Machine GPUs_DeviceName \
    GPUs_GlobalMemoryMb GPUs_Capability GPUs_DriverVersion
```
This gives TaskVine a real-time map of every GPU in the pool — its architecture, total VRAM, compute capability, driver version, and current availability. TaskVine then finds the best-fit GPU slot for the task's resource profile — not the largest available GPU, but the *smallest GPU that satisfies the task's requirements*, maximizing cluster-wide utilization by preserving large GPUs for large tasks.

* Phase 4: Precision ClassAd Construction and HTCondor Submission Using the matched GPU specification, TaskVine auto-generates an HTCondor **ClassAd** with exact, empirically-derived resource constraints:

```
universe = vanilla
request_GPUs = 1
request_memory = <peak_vram_mb + safety_margin> MB
requirements = (TARGET.GPUs_GlobalMemoryMb >= <peak_vram_mb>) && \
               (TARGET.GPUs_Capability >= <min_compute_capability>) && \
               (TARGET.GPUs_DeviceName == "<matched_gpu_model>")
```

This is fundamentally different from a user-written ClassAd, which is typically a rough guess. TaskVine's ClassAd is a **precision specification** grounded in observed runtime behavior, which means:

- The job will not OOM because VRAM is guaranteed to be sufficient
- The job will not be over-allocated because the resource request is tight to actual need
- The cluster scheduler can pack more jobs onto the same node because each job's footprint is accurately declared
- Large GPUs are preserved for tasks that actually need them

#### *Argument 2:*
##### GPU-Aware Data Locality and Model State Reuse in TaskVine
TaskVine's core design philosophy revolves around **caching and reusing intermediate outputs** at the worker level — if a downstream task needs data that a prior task already produced on a given worker, TaskVine avoids redundant transfers by keeping that data local and routing the dependent task to the same worker. This is a well-established throughput optimization for CPU-bound workflows. The proposal here is to extend this locality principle into the **GPU memory space**, where the cost of data movement is even more severe.

***NOTE*: Thanh's work so far assumes task A and task B use the same LLM but do inferences on different inputs. The LLM then is loaded once and shared by both task A and B so they can do inferences on the same LLM, instead of each task creating their own LLM and destroying it at the end of their executions.**

* Scenario: Distributed Iterative Model Training, Consider a training pipeline where a single model is being trained collaboratively across a sequence of tasks. The workflow looks like this:
- A **preprocessing task** runs upstream and produces a prepared data batch — cleaned, tokenized, normalized, and ready for consumption
- **Task A** receives this batch, loads it onto GPU-X on Worker-W, and runs several training iterations, updating the model's internal parameters (weights, optimizer state, gradient accumulators) entirely within GPU-X's VRAM
- Meanwhile, the preprocessing pipeline has produced the **next batch**, and **Task B** is now ready to be scheduled

At this point, a naive scheduler — which is what TaskVine currently behaves as for GPU tasks — would treat Task B as independent. It would schedule it to whichever worker has an available GPU slot, potentially a completely different machine. This is precisely the inefficiency that GPU-aware locality scheduling would eliminate.

* The Proposed Behavior: Forced GPU Affinity for Stateful Tasks, Instead TaskVine should recognize that Task B is similar to Task A — not just a data dependency (the new batch), but a **GPU memory state dependency** (the live model parameters that Task A left resident on GPU-X). The scheduler should:
1. **Pin Task B to Worker-W, GPU-X** — the same worker and same physical GPU that Task A ran on, without preempting or offloading the GPU memory state between tasks
2. **Keep the model live in VRAM** across the task boundary — Task A's training state (model weights + optimizer state) remains allocated on GPU-X, and Task B simply resumes training from that state with the new batch, paying zero reload cost
3. **Treat the GPU memory footprint of the model as a persistent worker resource** — analogous to how TaskVine treats cached files on disk, but for GPU VRAM

* The result is that the sequence of training tasks behaves like a **single continuous training process** from the GPU's perspective, even though TaskVine is orchestrating it as discrete tasks. 
* The Aggregation Step: Periodic Parameter Offload This is not an infinite loop on one GPU. After a configured number of training iterations — say every K tasks — the scheduler deliberately triggers a **checkpoint and offload phase**:
- The model parameters are copied from GPU VRAM to CPU, serialized, and sent to a **parameter aggregation step** (analogous to a reduce step in federated learning or parameter server architectures)
- Multiple workers, each of which has been training their own sequence of tasks on their own GPU using different data shards, contribute their parameter updates
- The aggregated parameters are then redistributed back to all workers, and the next round of training tasks begins


### Expermients that proves the arguments

### Some useful info
* Heterogeneous GPU communication approaches

|Transfer Method|Path                 |Relative Speed |Heterogeneous Support   |
|---------------|---------------------|---------------|------------------------|
|NVLink         |Direct GPU-to-GPU    |High (900 GB/s)|Poor (Requires same gen)|
|PCIe P2P       |GPU → Switch → GPU   |Medium (32-64 GB/s)|Good (Standard)    |
|Host-Mediated  |GPU → RAM → GPU      | Low (Variable)    |Universal (Default)|
