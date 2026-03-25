
### Argument
TasVine needs an internal GPU aware monitoring system for better allocation of GPU resources and better utilization of GPU resources for better running collocated jobs on HTCondor or SLURM.
#### *Argument 1:*
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