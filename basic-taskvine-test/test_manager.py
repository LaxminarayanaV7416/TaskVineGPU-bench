import os
import ndcctools.taskvine as vine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create a new manager
manager = vine.Manager([9123, 9129], name = "resnet_trainer")
print(f"Listening on port {manager.port}")

python_program = manager.declare_file(f"{BASE_DIR}/test.py", cache="workflow")

# Submit several tasks using that file.
print("Submitting tasks...")

task = vine.Task("python test.py")
task.add_input(python_program, "test.py")
task.set_cores(2)
task.set_memory(4096)
manager.submit(task)


# As they complete, display the results:
print("Waiting for tasks to complete...")
while not manager.empty():
    task = manager.wait(5)
    if task:
        print(f"Task {task.id} completed with result {task.output}")

print("All tasks done.")
