#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <dlfcn.h>
#include <stdlib.h>

// ========================= NVML Header content =================
typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25,
    NVML_ERROR_DEPRECATED = 26,
    NVML_ERROR_NOT_READY = 27,
    NVML_ERROR_GPU_NOT_FOUND = 28,
    NVML_ERROR_INVALID_STATE = 29,
    NVML_ERROR_RESET_TYPE_NOT_SUPPORTED = 30,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef void * nvmlDevice_t;
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
}  nvmlUtilization_t;

struct nvml_library {
    // library handle is tucked in the struct
    void *lib_handle;

    // all nvml function mocks
    nvmlReturn_t (*nvmlInit)(void);
    nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, nvmlDevice_t *);
    char *(*nvmlErrorString)(nvmlReturn_t);
    nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
    nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
    nvmlReturn_t (*nvmlShutdown)(void);
};

// ==============================================================

const char * LIBRARY_SEARCH_COMMAND = "ldconfig -p | grep %s | awk '{print $NF}'";

struct library_search_result {
    char path[512];
    bool found;
};

struct library_search_result find_library_by_name(const char *lib_name) {
    char command[256];
    char result[512];
    struct library_search_result lib_result = {{0}, false};

    snprintf(command, sizeof(command), LIBRARY_SEARCH_COMMAND, lib_name);

    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        perror("popen failed");
        return lib_result;
    }

    if (fgets(result, sizeof(result), fp) != NULL) {
        result[strcspn(result, "\n")] = 0;
        strncpy(lib_result.path, result, sizeof(lib_result.path) - 1);
        lib_result.path[sizeof(lib_result.path) - 1] = '\0';
        lib_result.found = true;
        printf("Library found at path %s\n", lib_result.path);
    } else {
        printf("Library %s not found in system search paths.\n", lib_name);
    }

    pclose(fp);
    return lib_result;
}

void nvml_library_close(struct nvml_library *lib){
    printf("We are shutting down NVML\n");
    if (!lib) {
        return;
    }
    if (lib->lib_handle) {
        dlclose(lib->lib_handle);
    }
    free(lib);
}

struct nvml_library *nvml_library_open(struct library_search_result res) {
    // store the nvml_library on heap
    struct nvml_library *nvml;
    nvml = malloc(sizeof(*nvml));

    if (!res.found || res.path[0] == '\0') {
        fprintf(stderr, "Error: Library path was not found.\n");
        return nvml;
    }

    printf("We are going to open the library from path %s\n", res.path);
    nvml->lib_handle = dlopen(res.path, RTLD_LAZY);
    if (!nvml->lib_handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        nvml_library_close(nvml);
        dlclose(nvml->lib_handle);
        return NULL;
    }
    printf("library opening completed\n");

    // instantiating the init call
    nvml->nvmlInit = (nvmlReturn_t (*)(void))dlsym(nvml->lib_handle, "nvmlInit");
    if (!nvml->nvmlInit) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlInit completed!\n");

    nvml->nvmlDeviceGetHandleByIndex = (nvmlReturn_t (*)(int, nvmlDevice_t *))dlsym(nvml->lib_handle, "nvmlDeviceGetHandleByIndex");
    if (!nvml->nvmlDeviceGetHandleByIndex) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlDeviceGetHandleByIndex completed!\n");

    nvml->nvmlErrorString = (char * (*)(nvmlReturn_t))dlsym(nvml->lib_handle, "nvmlErrorString");
    if (!nvml->nvmlErrorString) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlErrorString completed!\n");

    // nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
    nvml->nvmlDeviceGetMemoryInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *))dlsym(nvml->lib_handle, "nvmlDeviceGetMemoryInfo");
    if (!nvml->nvmlDeviceGetMemoryInfo) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlDeviceGetMemoryInfo completed!\n");

    // nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
    nvml->nvmlDeviceGetUtilizationRates = (nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t *))dlsym(nvml->lib_handle, "nvmlDeviceGetUtilizationRates");
    if (!nvml->nvmlDeviceGetUtilizationRates) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlDeviceGetUtilizationRates completed!\n");

    // nvmlReturn_t (*nvmlShutdown)(void);
    nvml->nvmlShutdown = (nvmlReturn_t (*)(void))dlsym(nvml->lib_handle, "nvmlShutdown");
    if (!nvml->nvmlShutdown) {
        fprintf(stderr, "Function not found: %s\n", dlerror());
    }
    printf("library nvmlShutdown completed!\n");
    return nvml;
}

void get_gpu_memory_usage(int device_id, struct nvml_library * nvml_lib){
    nvmlReturn_t init_result = nvml_lib->nvmlInit();
    printf("initializtion result %d\n",init_result);
    if (init_result != NVML_SUCCESS) {
        return;
    }

    nvmlDevice_t device;
    nvmlReturn_t result = nvml_lib->nvmlDeviceGetHandleByIndex(device_id, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle: %s\n", nvml_lib->nvmlErrorString(result));
        return;
    }
    nvmlMemory_t memory;
    result = nvml_lib->nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
        printf("Failed to get memory info: %s\n", nvml_lib->nvmlErrorString(result));
        return;
    }
    printf("GPU Memory Usage: %llu / %llu MB\n", memory.used / 1024 / 1024, memory.total / 1024 / 1024);

    nvmlReturn_t shutdown_result = nvml_lib->nvmlShutdown();
    if (shutdown_result != NVML_SUCCESS) {
        return;
    }
}

int main() {
    // Search for NVML without linking or loading it
    struct library_search_result res = find_library_by_name("libnvidia-ml.so");
    // struct library_search_result res = {.found = true, .path="/usr/lib/libnvidia-ml.so"};
    struct nvml_library *nvml_lib;

    if (res.found) {
        nvml_lib = nvml_library_open(res);
        get_gpu_memory_usage(0, nvml_lib);
    }

    // close the lib and free the memory
    nvml_library_close(nvml_lib);
    return 0;
}
