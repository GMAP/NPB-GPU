# NVIDIA GPU Drivers + CUDA

# Sources
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

# Remove old installations
```
sudo apt purge 'nvidia*'
sudo apt autoremove
```

# Block nouveau to not conflict with AMD or Intel
```
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
```

# Download the installer
```
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run
chmod +x cuda_13.0.1_580.82.07_linux.run
```

# Install only the NVIDIA driver (without interfering with AMD and without touching X)
```
sudo sh cuda_13.0.1_580.82.07_linux.run --silent --driver --dkms --no-drm --no-x-check
```

# Install only the CUDA Toolkit and examples
```
sudo sh cuda_13.0.1_580.82.07_linux.run --silent --toolkit --samples --no-driver
```

# Configure environment variables
```
mkdir -p ~/env
cat << 'EOF' > ~/env/nvidia-cuda.sh
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
EOF
```

# Compile and run a CUDA program with NVIDIA
1. Load nvidia-cuda environment variables
```
source ~/env/nvidia-cuda.sh
nvcc --version
```

2. Create the hello.cu program:
```c++
#include <cuda.h>
#include <iostream>
__global__ void hello(){
  printf("Hello World!\n");
}
int main(){
  hello<<<1,1>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

3. Compile and run the hello.cu program:
```
nvcc hello.cu -o hello
./hello
```