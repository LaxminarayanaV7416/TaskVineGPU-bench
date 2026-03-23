# TaskVineGPU-bench
This repository consists of TaskVine Application where we are designing the stress test for running multiple inference jobs as single task on taskvine where each inference jobs will eventually use the GPU

#### Experiment Goals
-----------------------
Training of the Model and inference of the model
1. we will declare a file which consists of multiple DeepLearning models and each model will be run as a separate task on TaskVine (lets aim to be realistic and create a serverless function and also Task both in seperate experiments and also might include )

### Understanding pieces of TaskVine so far:
-------------------------
1. TaskVine creates the standalone sandbox environment basically using the file system, where each new task gets executed on the worker creates its own isolated directory, this makes it easy to run multiple tasks in parallel without any interference between them.
2. We can actually set the number of cores, GPUs and memory per task but basically what is happening in behind the scenes is that mental model is setup and worker reports manager that it has available resources and manager assigns the task to the worker there is no real acquisition of resources is happening. The question I get immediately is what happens when the task is assigned to a worker goes out of memory, in that case the task will return something called as VINE_RESULT_RESOURCE_EXHAUSTED and if the resource monitoring is enabled it will report the peak memory usage to the manager.
3. Workers will timeout eventually after 15 minutes of inactivity, we can actually set this time using the `-t` option when starting the worker.

### Design ideas of the GPU scheduler:
-------------------------
- Look at the design of hugging face accelerator it basically has knowledge of the available GPUs and assigns tasks to them based on the available resources, by offloading the unrequired layers to the CPU and disk and only keeping the required layers in GPU memory.


#### resource monitoring using nvidia-smi
--------------
* we can use the `nvidia-smi --query-gpu=` command to monitor the GPU resources, including memory usage, GPU utilization, and temperature. (use the `--help-query-gpu` option to see the available fields)
*


#### Corrections in the documentation
--------------
1. Functional Abstractions example is showing wrong output
2. Futures section have one missing python markdown code block

#### Some useful links
--------------------
1. https://github.com/huggingface/trl/issues/2262
2. 