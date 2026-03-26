"""
This program occupies both fixed and percentage of the GPU memory and keeps the GPU busy.

NOTE: wiered behavior of GPU memory allocation
when we say ```torch.cuda.is_available()```
CUDA initializes a context, which includes:
- Driver state
- Kernel modules
- cuBLAS, cuDNN, etc.
The above alone can consume hundreds of MB

PyTorch doesn't just allocate exactly what you request. Instead, it uses a caching allocator:
- Requests memory in larger chunks
- Keeps unused memory cached for reuse
- Avoids expensive cudaMalloc calls
So, you request 1 GB, PyTorch may reserve ~1.5–2 GB internally

"""

import time
from enum import StrEnum

import torch


class SaturationModeEnum(StrEnum):
    PERCENTAGE = "percentage"
    FIXED = "fixed"


# --- Config ---
TARGET_UTILIZATION = 0.30  # 30% of GPU memory
FIXED_SATURATION_RATE_MB = 1024  # total of only 1GB
MODE = SaturationModeEnum.FIXED

# Select GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    target_mem = 0
    if MODE == SaturationModeEnum.PERCENTAGE:
        # Get total GPU memory
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory
        print(f"Total GPU memory: {total_mem / 1e9:.2f} GB")
        # Compute how much memory to allocate
        target_mem = int(total_mem * TARGET_UTILIZATION)
        print(
            f"Allocating: {target_mem / 1e9:.2f} GB (~{TARGET_UTILIZATION * 100:.0f}%)"
        )
    elif MODE == SaturationModeEnum.FIXED:
        target_mem = int(FIXED_SATURATION_RATE_MB * (1024**2))
        print(f"Allocating: {target_mem / 1e9:.2f} GB")

    if target_mem > 0:
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
                print(
                    f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
                )

        except KeyboardInterrupt:
            print("Stopping...")
else:
    print("CUDA is not available. its waste of saturating the CPU so good bye!!.")
