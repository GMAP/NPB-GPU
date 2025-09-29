# AMD GPU Drivers + ROCm + HIP

# Sources
- [AMD Linux Drivers](https://www.amd.com/en/support/download/linux-drivers.html#linux-for-radeon-pro)
- [ROCm Installation Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#rocm-installation)

# Remove old installations
```
sudo apt remove --purge 'rocm-*' 'hip-*' 'amdgpu-*'
sudo apt autoremove
sudo rm -rf /opt/rocm*
```

# AMD GPU driver installation
Ubuntu 22.04:
```
wget https://repo.radeon.com/amdgpu-install/7.0.1/ubuntu/jammy/amdgpu-install_7.0.1.70001-1_all.deb
sudo apt install ./amdgpu-install_7.0.1.70001-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
```

Ubuntu 24.04:
```
wget https://repo.radeon.com/amdgpu-install/7.0.1/ubuntu/noble/amdgpu-install_7.0.1.70001-1_all.deb
sudo apt install ./amdgpu-install_7.0.1.70001-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
```

# ROCm installation
Ubuntu 22.04:
```
wget https://repo.radeon.com/amdgpu-install/7.0.1/ubuntu/jammy/amdgpu-install_7.0.1.70001-1_all.deb
sudo apt install ./amdgpu-install_7.0.1.70001-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```

Ubuntu 24.04:
```
wget https://repo.radeon.com/amdgpu-install/7.0.1/ubuntu/noble/amdgpu-install_7.0.1.70001-1_all.deb
sudo apt install ./amdgpu-install_7.0.1.70001-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```

# Configure environment variables
```
mkdir -p ~/env
cat << 'EOF' > ~/env/amd-hip.sh
export HIP_PATH=/opt/rocm
export PATH=$HIP_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$HIP_PATH/include:$CPLUS_INCLUDE_PATH
export HIP_PLATFORM=amd
export HIP_RUNTIME=rocclr
EOF
```

# Compile and run a HIP program with AMD
1. Load amd-hip environment variables
```
source ~/env/amd-hip.sh
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