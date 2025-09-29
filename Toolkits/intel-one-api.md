# Intel GPU Drivers + one API

# Sources
- [Intel Compute Runtime](https://github.com/intel/compute-runtime/releases)
- [Intel GPU Drivers](https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)
- [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html)

# Installation procedure on Ubuntu 24.04
1. Create temporary directory

```
mkdir neo
```

2. Download all *.deb packages (you can replace the files with ones from the latest release)
```
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-core-2_2.18.5+19820_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-opencl-2_2.18.5+19820_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc_25.35.35096.9-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd_25.35.35096.9-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libigdgmm12_22.8.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1_25.35.35096.9-0_amd64.deb
```

3. Install all packages as root and fix any broken dependencies
```
sudo dpkg -i *.deb
sudo apt --fix-broken install
sudo dpkg -i *.deb

```

4. Download the one API HPC toolkit
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2d2a6686-ff06-44ce-baf0-ab84f8dafa89/intel-oneapi-hpc-toolkit-2025.2.1.44_offline.sh
```

5. Install the HPC toolkit
```
chmod +x intel-oneapi-hpc-toolkit-2025.2.1.44_offline.sh
sudo sh ./intel-oneapi-hpc-toolkit-2025.2.1.44_offline.sh -a --silent --cli --eula accept
```

### Compile and run a level zero program with Intel
1. Load intel-hip environment variables
```
source ~/intel/oneapi/setvars.sh --force
```

2. Create the hello.cl program:
```c++
__kernel void hello(__global int* result) {
    // writes 70x7 to the output
    result[0] = 70*7;
}
```

3. Create the hello.c program:
```c++
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    // Get drivers
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, NULL);
    ze_driver_handle_t* drivers = (ze_driver_handle_t*) malloc(sizeof(ze_driver_handle_t) * driverCount);
    zeDriverGet(&driverCount, drivers);

    // Get devices
    uint32_t deviceCount = 0;
    zeDeviceGet(drivers[0], &deviceCount, NULL);
    ze_device_handle_t* devices = (ze_device_handle_t*) malloc(sizeof(ze_device_handle_t) * deviceCount);
    zeDeviceGet(drivers[0], &deviceCount, devices);

    // Create context
    ze_context_handle_t context;
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    zeContextCreate(drivers[0], &contextDesc, &context);

    // Create command queue
    ze_command_queue_desc_t queueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL,
                                         0, 0, 0, 0,
                                         ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                         ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_handle_t cmdQueue;
    zeCommandQueueCreate(context, devices[0], &queueDesc, &cmdQueue);

    // Create command list
    ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0};
    ze_command_list_handle_t cmdList;
    zeCommandListCreate(context, devices[0], &cmdListDesc, &cmdList);

    // Allocate memory on device
    int host_result = 0;
    ze_device_mem_alloc_desc_t deviceDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0};
    void* device_result;
    zeMemAllocDevice(context, &deviceDesc, sizeof(int), 1, devices[0], &device_result);

    // Load SPIR-V module
    FILE* file = fopen("hello_kernel.spv", "rb");
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void* spv = malloc(size);
    fread(spv, 1, size, file);
    fclose(file);

    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.pInputModule = spv;
    moduleDesc.inputSize = size;

    ze_module_handle_t module;
    zeModuleCreate(context, devices[0], &moduleDesc, &module, NULL);

    // Create kernel
    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernelDesc.pKernelName = "hello";
    ze_kernel_handle_t kernel;
    zeKernelCreate(module, &kernelDesc, &kernel);

    zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &device_result);

    // Launch kernel
    ze_group_count_t groupCount = {1,1,1};
    zeCommandListAppendLaunchKernel(cmdList, kernel, &groupCount, NULL, 0, NULL);

    // Copy result back to host
    zeCommandListAppendMemoryCopy(cmdList, &host_result, device_result, sizeof(int), NULL, 0, NULL);

    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, NULL);
    zeCommandQueueSynchronize(cmdQueue, UINT64_MAX);

    printf("Hello World from Level Zero GPU! result = %d\n", host_result);

    // Cleanup
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeCommandListDestroy(cmdList);
    zeCommandQueueDestroy(cmdQueue);
    zeMemFree(context, device_result);
    zeContextDestroy(context);

    free(spv);
    free(drivers);
    free(devices);

    return 0;
}

```

4. Create the list_devices.c program, which detects Intel and only Intel GPUs on the system:
```c++
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    // Get driver count
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, NULL);

    if (driverCount == 0) {
        printf("No Level Zero drivers found.\n");
        return 1;
    }

    // Allocate memory for drivers
    ze_driver_handle_t* drivers = (ze_driver_handle_t*) malloc(sizeof(ze_driver_handle_t) * driverCount);
    zeDriverGet(&driverCount, drivers);

    for (uint32_t d = 0; d < driverCount; d++) {
        // Get device count for this driver
        uint32_t deviceCount = 0;
        zeDeviceGet(drivers[d], &deviceCount, NULL);

        if (deviceCount == 0) {
            printf("No devices found for driver %u.\n", d);
            continue;
        }

        // Allocate memory for devices
        ze_device_handle_t* devices = (ze_device_handle_t*) malloc(sizeof(ze_device_handle_t) * deviceCount);
        zeDeviceGet(drivers[d], &deviceCount, devices);

        for (uint32_t i = 0; i < deviceCount; i++) {
            ze_device_properties_t props;
            zeDeviceGetProperties(devices[i], &props);
            printf("Driver[%u] Device[%u]: %s\n", d, i, props.name);
        }

        free(devices);
    }

    free(drivers);
    return 0;
}

```

5. Compile and run the hello.c program:
```
gcc list_devices.c -o list_devices -I$ONEAPI_ROOT/compiler/latest/linux/include -L$ONEAPI_ROOT/compiler/latest/linux/lib -lze_loader

# get GPU device list from Intel
./list_devices

# Replace B580 with your Intel GPU device detected on your system (through list_devices)
ocloc -file hello.cl -device B580 -output hello_kernel.spv

gcc hello.c -o hello_host -I$ONEAPI_ROOT/compiler/latest/linux/include -L$ONEAPI_ROOT/compiler/latest/linux/lib -lze_loader

./hello_host
```