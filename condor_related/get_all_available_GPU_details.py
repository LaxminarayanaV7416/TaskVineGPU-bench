import re
import subprocess

"""
GPUs = 3
GPUs_Capability = 7.0
GPUs_ClockMhz = 1455.0
GPUs_ComputeUnits = 80
GPUs_CoresPerCU = 64
GPUs_DeviceName = "NVIDIA TITAN V"
GPUs_DriverVersion = 13.0
GPUs_ECCEnabled = false
GPUs_GlobalMemoryMb = 12050
GPUs_GPU_32235e0a = [ Id = "GPU-32235e0a"; ClockMhz = 1455.0; Capability = 7.0; CoresPerCU = 64; DeviceName = "NVIDIA TITAN V"; DeviceUuid = "32235e0a-784d-a422-e6df-726e913aa35d"; ECCEnabled = false; ComputeUnits = 80; DriverVersion = 13.0; DevicePciBusId = "0000:3B:00.0"; GlobalMemoryMb = 12050; MaxSupportedVersion = 13000 ]
GPUs_GPU_487e38b2 = [ Id = "GPU-487e38b2"; ClockMhz = 1455.0; Capability = 7.0; CoresPerCU = 64; DeviceName = "NVIDIA TITAN V"; DeviceUuid = "487e38b2-4341-5424-b70e-f20f7787afb1"; ECCEnabled = false; ComputeUnits = 80; DriverVersion = 13.0; DevicePciBusId = "0000:86:00.0"; GlobalMemoryMb = 12050; MaxSupportedVersion = 13000 ]
GPUs_GPU_ef7999e4 = [ Id = "GPU-ef7999e4"; ClockMhz = 1455.0; Capability = 7.0; CoresPerCU = 64; DeviceName = "NVIDIA TITAN V"; DeviceUuid = "ef7999e4-b8a6-7d20-d2e9-590c9c9ac77c"; ECCEnabled = false; ComputeUnits = 80; DriverVersion = 13.0; DevicePciBusId = "0000:D8:00.0"; GlobalMemoryMb = 12050; MaxSupportedVersion = 13000 ]
GPUs_MaxSupportedVersion = 13000
GPUsMemoryUsage = 242.0
Machine = "ta-titanv-001.crc.nd.edu"
"""


def parse_gpu_string(gpu_string):
    """
    Parse a GPU configuration string into a dictionary.

    Args:
        gpu_string: String in the format "GPUs_GPU_32235e0a = [ ... ]"

    Returns:
        Dictionary containing the parsed GPU configuration
    """
    # Extract the part inside brackets
    match = re.search(r"=\s*\[(.*?)\]", gpu_string, re.DOTALL)
    if not match:
        return {}

    content = match.group(1)

    # Parse key-value pairs
    gpu_dict = {}

    # Find all key-value pairs (handles strings, numbers, booleans)
    pattern = r'(\w+)\s*=\s*(".*?"|\d+\.\d+|\d+|true|false)'

    for key, value in re.findall(pattern, content):
        # Parse the value based on its type
        if value.startswith('"') and value.endswith('"'):
            # String value - remove quotes
            gpu_dict[key] = value[1:-1]
        elif value == "true":
            gpu_dict[key] = True
        elif value == "false":
            gpu_dict[key] = False
        elif "." in value:
            # Float value
            gpu_dict[key] = float(value)
        else:
            # Integer value
            gpu_dict[key] = int(value)

    return gpu_dict


def parse_multiple_gpu_strings(text):
    """
    Parse multiple GPU configuration strings from text.

    Args:
        text: String containing one or more GPU configurations

    Returns:
        Dictionary with GPU IDs as keys and their configurations as values
    """
    gpus = {}

    # Find all GPU definitions
    pattern = r"GPUs_GPU_(\w+)\s*=\s*\[.*?\]"

    for match in re.finditer(pattern, text, re.DOTALL):
        gpu_id = match.group(1)
        full_string = match.group(0)
        gpus[gpu_id] = parse_gpu_string(full_string)

    return gpus


def parse_machine_hostname(text):
    """
    Parse Machine string and extract the hostname.

    Args:
        text: String containing Machine = "hostname"

    Returns:
        Hostname string or None if not found
    """
    # Regular expression to match Machine = "hostname"
    # Captures the hostname between quotes
    pattern = r'Machine\s*=\s*"([^"]+)"'

    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def get_gpu_info(x):
    # Parse multiple GPUs
    multiple_result = parse_multiple_gpu_strings(x)
    hostname = parse_machine_hostname(x)
    print(f"Hostname: {hostname}")
    print("Multiple GPUs parse result:")
    for gpu_id, config in multiple_result.items():
        print(f"GPU ID: {gpu_id}")
        print(f"  Config: {config}")


def get_all_available_gpu_details():
    result = subprocess.run(
        ["condor_status", "-constraint", "GPUs > 0", "-long"],
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    get_gpu_info(text)


if __name__ == "__main__":
    get_all_available_gpu_details()
