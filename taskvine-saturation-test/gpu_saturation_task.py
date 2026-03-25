"""
This program occupies 30% of the GPU memory and keeps the GPU busy.
"""

import time

import torch

# --- Config ---
TARGET_UTILIZATION = 0.30  # 30% of GPU memory

# Select GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    # Get total GPU memory
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory

    # Compute how much memory to allocate
    target_mem = int(total_mem * TARGET_UTILIZATION)

    print(f"Total GPU memory: {total_mem / 1e9:.2f} GB")
    print(f"Allocating: {target_mem / 1e9:.2f} GB (~{TARGET_UTILIZATION * 100:.0f}%)")

    # Allocate tensor (float32 = 4 bytes per element)
    num_elements = target_mem // 4
    tensor = torch.empty(num_elements, dtype=torch.float32, device=device)

    # Initialize it so memory is actually used
    tensor.fill_(1.0)

    print("Memory allocated. Entering compute loop...")

    # Keep GPU active with a simple loop
    try:
        while True:
            # simple operation to keep GPU busy
            tensor = tensor * 1.0000001

            # small sleep to avoid 100% utilization (optional)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
else:
    print("CUDA is not available. its waste of saturating the CPU so good bye!!.")
