#!/bin/bash

# Check if the nvidia-smi command is available
if ! command -v nvidia-smi &> /dev/null; then
  echo "error; nvidia-smi not found. Please make sure the NVIDIA driver is installed."
  exit 1
fi

# Check available NVIDIA GPUs
num_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)

# Exit if there is no NVIDIA GPUs
if [ "$num_gpus" -eq 0 ]; then
  echo "error; There is no NVIDIA GPUs available."
  exit 1
fi

# Read the GPU id from the gpu.config file (GPU_DEVICE variable)
gpu_id=$(grep "GPU_DEVICE" ../config/gpu.config | cut -d '=' -f 2 | tr -d ' ')

# If the gpu_id is not valid, set it to 0
if [ -z "$gpu_id" ] || [ "$gpu_id" -lt 0 ] || [ "$gpu_id" -ge "$num_gpus" ]; then
  echo "The GPU id $gpu_id is not valid. Using GPU 0 as default."
  gpu_id=0
fi

# Get the compute capability from the GPU specified
COMPUTE_CAPABILITY=$(nvidia-smi -i $gpu_id --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1)

# Extract the major and minor version of the compute capability
COMPUTE_MAJOR=$(echo $COMPUTE_CAPABILITY | cut -d '.' -f 1)
COMPUTE_MINOR=$(echo $COMPUTE_CAPABILITY | cut -d '.' -f 2)

# Construct the -gencode arch and code flags
GENCODE="-gencode arch=compute_${COMPUTE_MAJOR}${COMPUTE_MINOR},code=sm_${COMPUTE_MAJOR}${COMPUTE_MINOR}"

# Replace the COMPUTE_CAPABILITY line in the make.def file
sed -i "s|^COMPUTE_CAPABILITY.*|COMPUTE_CAPABILITY = ${GENCODE}|" ../config/make.def

echo "Updated COMPUTE_CAPABILITY (GPU of id ${gpu_id}) in ../config/make.def to ${GENCODE}"

exit 0