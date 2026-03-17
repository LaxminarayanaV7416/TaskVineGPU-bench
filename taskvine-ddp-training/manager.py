import os
import ndcctools.taskvine as vine
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import pandas as pd

def index_splitter(dataset: Dataset, shards = 1):
    n = len(dataset)
    indices = torch.randperm(n)
    split_size = n // shards
    df = pd.DataFrame()
    for i in range(shards):
        df[f"{i}"] = indices[i*split_size:(i+1)*split_size].tolist()
    return df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# lets download the MNIST dataset
train_dataset = datasets.MNIST(
    root=f"{BASE_DIR}/data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(root=f"{BASE_DIR}/data", train=False, transform=transforms.ToTensor())

# Create a new manager
manager = vine.Manager([9123, 9129], name = "resnet_trainer")
print(f"Listening on port {manager.port}")

# Declare a common input file to be shared by multiple tasks.
data = manager.declare_file(f"{BASE_DIR}/data", cache="forever")

# declare the package and its input file
poncho_file = manager.declare_file(f"{BASE_DIR}/package.tar.gz", cache="workflow")
poncho_pkg = manager.declare_poncho(poncho_file, cache="workflow")

python_program = manager.declare_file(f"{BASE_DIR}/resnet.py", cache="workflow")

NUMBER_OF_TASKS = 1
INDEX_FILE_NAME = "indices.csv"


df = index_splitter(train_dataset, shards=NUMBER_OF_TASKS)
df.to_csv(f"{BASE_DIR}/indices.csv", index=False)

indices = manager.declare_file(f"{BASE_DIR}/{INDEX_FILE_NAME}", cache="workflow")


# Submit several tasks using that file.
print("Submitting tasks...")
for index in range(NUMBER_OF_TASKS):
    task = vine.Task(f"python resnet.py --index={index} --indices-file={INDEX_FILE_NAME}")
    task.add_input(python_program, "resnet.py")
    task.add_input(indices, INDEX_FILE_NAME)
    task.add_input(data, "data")
    # attach the package to the task
    task.add_poncho_package(poncho_pkg)
    task.set_cores(2)
    task.set_memory(4096)
    task.set_gpus(1)
    task.add_feature("NVIDIA RTX A1000 Laptop GPU")
    manager.submit(task)


# As they complete, display the results:
print("Waiting for tasks to complete...")
while not manager.empty():
    task = manager.wait(5)
    if task:
        print(f"Task {task.id} completed with result {task.output}")

print("All tasks done.")
