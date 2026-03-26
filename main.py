import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
props = torch.cuda.get_device_properties(device)
print(props)
print("============")
print(props.total_memory)