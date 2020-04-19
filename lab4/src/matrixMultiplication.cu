/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 * Matrix multiplication C=A*B
*/

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <assert.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std::chrono;

__global__ void matrixMultiplication(const float *A, const float *B, float *C, int size) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(rowIdx < size && colIdx < size){
        float result = 0.;
        for(int i = 0; i < size; i++){
            result += A[rowIdx * size + i] * B[i * size + colIdx];
        }
        C[rowIdx * size + colIdx] = result;
    }
}


// Check if it works 
__global__ void strideMatrixMultiplication(const float *A, const float*B, const float*C, int size){
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    for(int i = colIdx; i < size; i += strideX){
        for(int j = rowIdx; j < size; j += strideY){
            float result = 0.;
            for(int k = 0; k < size; k++){
                result += A[j * size + k] * B[k * size + i];
            }
            C[j * size + i] = result;
        }
    }
}

void initializeMatrix(float *A, unsigned long numberOfAllElements){
    for(int i = 0; i < numberOfAllElements; i++){
        A[i] = rand()/(float)RAND_MAX;
    }
}

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {


}