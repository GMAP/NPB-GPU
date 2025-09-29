# GPU Toolkits for NPB Experiments

This folder contains installation and setup guides for GPU drivers and programming toolkits needed to run the NAS Parallel Benchmarks (NPB) on different GPU platforms. The instructions cover Intel, AMD, and NVIDIA GPUs, as well as HIP, CUDA, and oneAPI frameworks.

## Contents

| File | Description |
|------|-------------|
| `intel-one-api.md` | Guide to install Intel GPU drivers and oneAPI HPC toolkit. Includes example Level Zero programs. |
| `intel-hip.md` | Guide to install HIP support for Intel GPUs (via CHIP-SPV). |
| `amd-hip.md` | Guide to install AMD ROCm drivers and HIP support for AMD GPUs. |
| `nvidia-cuda.md` | Guide to install NVIDIA GPU drivers and CUDA toolkit. |
| `nvidia-hip.md` | Guide to install HIP for NVIDIA GPUs (via ROCm HIP port). |

## Usage

1. Choose the GPU platform you have: Intel, AMD, or NVIDIA.
2. Follow the corresponding Markdown guide to install the drivers and toolkits.
3. Use the provided examples (if available) to verify that the GPU environment is correctly configured.
4. You can now compile and run the NAS Parallel Benchmarks (NPB) on your GPU.

## Listing the GPUs on the system
- **Getting all GPUs**
```
lspci | grep -i vga
```

- **Getting Intel GPUs**
```
clinfo | grep -i intel
```

- **Getting AMD GPUs**
```
/opt/rocm/bin/rocminfo
```

- **Getting NVIDIA GPUs**
```
nvidia-smi
```

## Notes

- The guides are tested on Linux systems (mostly Ubuntu).
- Make sure to follow the instructions carefully, especially when setting environment variables.