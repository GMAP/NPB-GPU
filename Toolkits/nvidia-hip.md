# HIP for NVIDIA GPUs

# Sources
- [HIP Repository on GitHub](https://github.com/ROCm/hip)

# Install required dependencies
```
sudo apt update
sudo apt install -y git cmake build-essential python3-pip
```

# Clone the HIP repository
```
git clone https://github.com/ROCm-Developer-Tools/HIP.git hip-nvcc
cd hip-nvcc
```

# Create build folder
```
mkdir -p build && cd build
```

# Configure CMake for CUDA compilation
```
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/hip-nvcc \
  -DHIP_PLATFORM=nvidia \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda # inform your path to the CUDA directory
```

# Compile and install
```
make -j$(nproc)
sudo make install
```

# Configure environment variables
```
mkdir -p ~/env
cat << 'EOF' > ~/env/nvidia-hip.sh
export HIP_PATH=/opt/hip-nvcc
export PATH=$HIP_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$HIP_PATH/include:$CPLUS_INCLUDE_PATH
export HIP_PLATFORM=nvidia
export HIP_COMPILER=nvcc
export HIP_RUNTIME=cuda
EOF

```

# Compile and run a HIP program with NVIDIA
1. Load nvidia-hip environment variables
```
source ~/env/nvidia-hip.sh
hipcc --version
```

2. Create the hello.cpp program:
```c++
#include <hip/hip_runtime.h>
#include <iostream>
__global__ void hello(){
  printf("Hello World!\n");
}
int main(){
  hello<<<1,1>>>();
  hipDeviceSynchronize();
  return 0;
}
```

3. Compile and run the hello.cpp program:
```
hipcc hello.cpp -o hello
./hello
```