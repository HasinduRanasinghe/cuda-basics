#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Matrix dimensions: A(M x K) * B(K x N) = C(M x N)
    int M = 512, N = 512, K = 512;
    
    // Host matrices
    std::vector<float> h_A(M * K, 1.0f);  
    std::vector<float> h_B(K * N, 2.0f);  
    std::vector<float> h_C(M * N, 0.0f); 
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    
    // Allocate GPU memory
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);
    
    matmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print a few results for verification
    std::cout << "Matrix multiplication completed!" << std::endl;
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Expected result
    std::cout << "Expected value: " << 2.0f * K << std::endl;
    
    // Output the time taken
    std::cout << "Time taken for matrix addition on GPU: " << milliseconds << " ms" << std::endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}