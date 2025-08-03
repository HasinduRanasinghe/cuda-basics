# CUDA Basics

This repository contains simple CUDA programs to demonstrates basic CUDA concepts such as kernel launches, memory management, and GPU timing. The included programs are,
1. Element-wise addition of two large square matrices.

## Features

### Matrix Addition
- Adds two `N x N` matrices (`N = 10000`) in parallel on the GPU. Modify the `#define N` value to change the size.
- Measures and reports the kernel execution time.
- Uses a 2D grid and block structure for thread organization.
- Handles memory allocation and data transfers between host (CPU) and device (GPU).

## Requirements
- **NVIDIA CUDA Toolkit** (version 12.4 or later recommended).
- **NVIDIA GPU** with CUDA support.
- **C++ Compiler** compatible with CUDA (`nvcc`).

## Installation
1. Install the NVIDIA CUDA Toolkit. Follow the instructions for your platform.
2. Ensure a compatible NVIDIA GPU driver is installed.
3. Clone this repository:
   ```bash
   git clone https://github.com/HasinduRanasinghe/cuda-basics.git
   cd cuda-basics
   ```

## Compilation
To compile the program, use the `nvcc` compiler included in the CUDA Toolkit. 
```bash
nvcc <file.cu> -o <obj_file_name>
```

## Usage
Run the compiled executable:
```bash
./<file>
```
The program will:
1. Initialize two 10000x10000 matrices with random float values.
2. Perform matrix addition on the GPU.
3. Output the execution time of the CUDA kernel in milliseconds.

Example output:
```
Non-optimized time taken for matrix addition on GPU: 12.345 ms
```

## Code Structure

### Matrix Addition
- **File**: `cuda_mat_add.cu`
- **Key Components**:
  - **Kernel**: `matrixAdd` - Performs element-wise matrix addition using a 2D thread grid.
  - **Main Function**:
    - Allocates host and device memory.
    - Initializes input matrices with random values.
    - Copies data to the GPU.
    - Launches the CUDA kernel with a 625x625 grid of 16x16 thread blocks.
    - Measures execution time using CUDA events.
    - Copies the result back to the host and frees memory.

## Notes
- The implementations are non-optimized for simplicity, focusing on demonstrating CUDA basics.
- Potential optimizations include using shared memory, coalesced memory access, or tuning block sizes.
- No explicit error checking is included. If needed, add checks for CUDA API calls (`cudaGetLastError`).
- The results is not verified for correctness.
