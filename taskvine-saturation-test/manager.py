import os

import ndcctools.taskvine as vine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create a new manager
manager = vine.Manager([9123, 9129], name="resnet_trainer")
print(f"Listening on port {manager.port}")

# declare the package and its input file
poncho_file = manager.declare_file(f"{BASE_DIR}/package.tar.gz", cache="workflow")
poncho_pkg = manager.declare_poncho(poncho_file, cache="workflow")

python_program = manager.declare_file(
    f"{BASE_DIR}/gpu_saturation_task.py", cache="workflow"
)

NUMBER_OF_TASKS = 100

# Submit several tasks using that file.
print("Submitting tasks...")
for index in range(NUMBER_OF_TASKS):
    task = vine.Task(f"python gpu_saturation_task.py")
    task.add_input(python_program, "gpu_saturation_task.py")
    # attach the package to the task
    task.add_poncho_package(poncho_pkg)
    task.set_cores(2)
    task.set_memory(1024)
    task.set_gpus(1)
    # task.add_feature("NVIDIA RTX A1000 Laptop GPU")
    manager.submit(task)


# As they complete, display the results:
print("Waiting for tasks to complete...")
while not manager.empty():
    task = manager.wait(5)
    if task:
        print(f"Task {task.id} completed with result {task.output}")

print("All tasks done.")
