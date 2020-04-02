/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace std::chrono;

static std::ofstream output;
// static ofstream file_1D2D = ofstream("file_1D2D.csv", std::iostream::out);
// static ofstream file_2D2D = ofstream("file_2D2D.csv", std::iostream::out);
// static ofstream file_CPU = ofstream("file_CPU.csv", std::iostream::out);
int MSBsqrt(const int n)
{
    int sn = sqrt(n);
    if( 0 != n*n%sn )
    {
        int counter = 1;
        int psn = sn;
        while(psn)
        {
            psn = (psn >> 1);
            counter++;
        }
        counter--;
        sn = 1;
        sn = (sn << counter);
    }
    return sn;
}

template<class T>
__global__ void printThreadIndex2D(const T* MatA, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d)"
           "global index %2d ival %2f\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, MatA[idx]);
}
template<class T>
__device__ void printThreadIndex1D(const T* MatC, const int ny, const int ntotal)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ix = idx%ny;
    int iy = idx/ny;

    if(idx < ntotal)
    {
        printf("thread_id (%d) block_id (%d) coordinate (%d,%d)"
        "global index %2d ival %.2f\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, MatC[idx]);
    }
}

template<class T>
__global__ void matrixAdd_1D1D(const float* MatA, const float* MatB, float* MatC, const int ny, const int ntotal)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int ix = idx%ny;
    // int iy = idx/ny;

    if(idx < ntotal)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
        #if 0
        printf("grid_dim (%d, %d, %d), block_dim (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z);
        printf("block_dim (%d) thread_id (%d) block_id (%d) coordinate (%d,%d)"
        "global index %2d ival %.2f\n", blockDim.x, threadIdx.x, blockIdx.x, ix, iy, idx, MatC[idx]);
        #endif
    }
}

template<class T>
__global__ void matrixAdd_2D2D(const T* MatA, const T* MatB, T* MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*ny + ix;

    if(ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
        #if 0
        printf("matC=%.2f, matA=%.2f, matB=%.2f\n", MatC[idx], MatA[idx], MatB[idx]);
        printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d)"
        "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, MatC[idx]);
        #endif
    }

    
}

template<class T>
__global__ void matrixAdd_2D1D(const T* MatA, const T* MatB, T* MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y*gridDim.x*blockDim.x;
    unsigned int idx = iy + ix;

    if( idx < nx*ny )
    {
        MatC[idx] = MatA[idx] + MatB[idx];
        #if 0
        printf("matC=%.2f, matA=%.2f, matB=%.2f\n", MatC[idx], MatA[idx], MatB[idx]);
        printf("thread_id (%d) block_id (%d,%d) coordinate (%d,%d)"
        "global index %2d ival %.2f\n", threadIdx.x, blockIdx.x, blockIdx.y, ix, iy, idx, MatC[idx]);
        #endif
    }
}

float* allocateFloatMatrix(const int nx, const int ny)
{
	float* handle = (float*)malloc(nx * ny * sizeof(float));

	return handle;
}

template<class T>
void fillMatrixRandom(T* matrix, const int nx, const int ny)
{
    for(int j = 0; j < nx; j++)
    {
    	for(int i = 0; i < ny; i++)
    	{
            matrix[j * nx + i] = rand()/(float)RAND_MAX;
        }
    }
}

template<>
void fillMatrixRandom<int>(int* matrix, const int nx, const int ny)
{
    for(int j = 0; j < nx; j++)
    {
    	for(int i = 0; i < ny; i++)
    	{
            matrix[j * nx + i] = rand()%100;
        }
    }
}

void exec_1D1D(const int nx, const int ny)
{
    cudaError_t err = cudaSuccess;

    size_t sizeOfAllocationOnGraphicsCard = nx*ny*sizeof(float);
    float* host_matrixA = allocateFloatMatrix(nx, ny);
    float* host_matrixB = allocateFloatMatrix(nx, ny);
    float* host_matrixC = allocateFloatMatrix(nx, ny);
    
    if (host_matrixA == NULL || host_matrixB == NULL || host_matrixC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }
    
    //device allocation
    float* dev_matrixA, *dev_matrixB, *dev_matrixC;
    if( cudaSuccess != (err = cudaMalloc((void**)&dev_matrixA, sizeOfAllocationOnGraphicsCard)) ||  
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixB, sizeOfAllocationOnGraphicsCard)) ||
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixC, sizeOfAllocationOnGraphicsCard)))
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //filling matrices with random values
    printf("Fill matrices\n");
    fillMatrixRandom(host_matrixA, nx, ny);
    fillMatrixRandom(host_matrixB, nx, ny);
    fillMatrixRandom(host_matrixC, nx, ny);

    // Error code to check return values for CUDA calls
    

    //copying data on graphics card
    if( cudaSuccess != (err = cudaMemcpy(dev_matrixA, host_matrixA, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)) ||
        cudaSuccess != (err = cudaMemcpy(dev_matrixB, host_matrixB, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)))
    {
        fprintf(stderr, "Failed to copy matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the matrixAdd CUDA Kernel
    int blocksPerGrid = 1;
    int threadsPerBlock = nx * ny;
    dim3 threadsInBlock(threadsPerBlock);
    printf("%d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    auto start = high_resolution_clock::now();
    matrixAdd_1D1D<float><<<blocksPerGrid, threadsInBlock>>>(dev_matrixA, dev_matrixB, dev_matrixC, ny, nx*ny);
    auto stop = high_resolution_clock::now();
    auto durationOnCUDA = duration_cast<microseconds>(stop - start);
    output << durationOnCUDA.count();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to launch matrixAdd kernel (error code %s)!\n", __func__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(host_matrixC, dev_matrixC, sizeOfAllocationOnGraphicsCard, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i <  nx * ny; ++i)
    {
        if (fabs(host_matrixA[i] + host_matrixB[i] - host_matrixC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_matrixC);

    printf("Done\n");
}

void exec_1D2D(const int nx, const int ny)
{
    cudaError_t err = cudaSuccess;

    size_t sizeOfAllocationOnGraphicsCard = nx*ny*sizeof(float);

    float* host_matrixA = allocateFloatMatrix(nx, ny);
    float* host_matrixB = allocateFloatMatrix(nx, ny);
    float* host_matrixC = allocateFloatMatrix(nx, ny);
    
    if (host_matrixA == NULL || host_matrixB == NULL || host_matrixC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }
    
    //device allocation
    float* dev_matrixA, *dev_matrixB, *dev_matrixC;
    if( cudaSuccess != (err = cudaMalloc((void**)&dev_matrixA, sizeOfAllocationOnGraphicsCard)) ||  
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixB, sizeOfAllocationOnGraphicsCard)) ||
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixC, sizeOfAllocationOnGraphicsCard)))
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //filling matrices with random values
    printf("Fill matrices\n");
    fillMatrixRandom(host_matrixA, nx, ny);
    fillMatrixRandom(host_matrixB, nx, ny);
    fillMatrixRandom(host_matrixC, nx, ny);

    // Error code to check return values for CUDA calls
    

    //copying data on graphics card
    if( cudaSuccess != (err = cudaMemcpy(dev_matrixA, host_matrixA, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)) ||
        cudaSuccess != (err = cudaMemcpy(dev_matrixB, host_matrixB, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)))
    {
        fprintf(stderr, "Failed to copy matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the matrixAdd CUDA Kernel
    int snx = MSBsqrt(nx);
    int sny = MSBsqrt(ny);
    dim3 blocksPerGrid(snx, sny);
    //int threadsPerBlock = nx * ny;
    dim3 threadsInBlock((nx*ny)/(snx*sny));
    printf("(%d,%d) blocks of (%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsInBlock.x);
    auto start = high_resolution_clock::now();
    matrixAdd_2D1D<float><<<blocksPerGrid, threadsInBlock>>>(dev_matrixA, dev_matrixB, dev_matrixC, nx, ny);
    auto stop = high_resolution_clock::now();
    auto durationOnCUDA = duration_cast<microseconds>(stop - start);
    output << durationOnCUDA.count();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to launch matrixAdd kernel (error code %s)!\n", __func__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(host_matrixC, dev_matrixC, sizeOfAllocationOnGraphicsCard, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i <  nx * ny; ++i)
    {
        if (fabs(host_matrixA[i] + host_matrixB[i] - host_matrixC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_matrixC);

    printf("Done\n");
}

void exec_2D2D(const int nx, const int ny)
{
    cudaError_t err = cudaSuccess;

    size_t sizeOfAllocationOnGraphicsCard = nx*ny*sizeof(float);

    float* host_matrixA = allocateFloatMatrix(nx, ny);
    float* host_matrixB = allocateFloatMatrix(nx, ny);
    float* host_matrixC = allocateFloatMatrix(nx, ny);
    
    if (host_matrixA == NULL || host_matrixB == NULL || host_matrixC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }
    
    //device allocation
    float* dev_matrixA, *dev_matrixB, *dev_matrixC;
    if( cudaSuccess != (err = cudaMalloc((void**)&dev_matrixA, sizeOfAllocationOnGraphicsCard)) ||  
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixB, sizeOfAllocationOnGraphicsCard)) ||
        cudaSuccess != (err = cudaMalloc((void**)&dev_matrixC, sizeOfAllocationOnGraphicsCard)))
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //filling matrices with random values
    printf("Fill matrices\n");
    fillMatrixRandom(host_matrixA, nx, ny);
    fillMatrixRandom(host_matrixB, nx, ny);
    fillMatrixRandom(host_matrixC, nx, ny);

    // Error code to check return values for CUDA calls
    

    //copying data on graphics card
    if( cudaSuccess != (err = cudaMemcpy(dev_matrixA, host_matrixA, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)) ||
        cudaSuccess != (err = cudaMemcpy(dev_matrixB, host_matrixB, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice)))
    {
        fprintf(stderr, "%s: Failed to copy matrix from host to device (error code %s)!\n", __func__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the matrixAdd CUDA Kernel
    int snx = MSBsqrt(nx);
    int sny = MSBsqrt(ny);
    dim3 blocksPerGrid(snx, sny);
    //int threadsPerBlock = nx * ny;
    dim3 threadsInBlock((nx/snx), (ny/sny));
    printf("(%d,%d) blocks of (%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsInBlock.x, threadsInBlock.y);
    auto start = high_resolution_clock::now();
    matrixAdd_2D2D<float><<<blocksPerGrid, threadsInBlock>>>(dev_matrixA, dev_matrixB, dev_matrixC, ny, ny);
    auto stop = high_resolution_clock::now();
    auto durationOnCUDA = duration_cast<microseconds>(stop - start);
    output << durationOnCUDA.count();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to launch matrixAdd kernel (error code %s)!\n", __func__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(host_matrixC, dev_matrixC, sizeOfAllocationOnGraphicsCard, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i <  nx * ny; ++i)
    {
        if (fabs(host_matrixA[i] + host_matrixB[i] - host_matrixC[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_matrixC);

    printf("Done\n");
}

void exec_CPU(const int nx, const int ny)
{
    float* host_matrixA = allocateFloatMatrix(nx, ny);
    float* host_matrixB = allocateFloatMatrix(nx, ny);
    float* host_matrixC = allocateFloatMatrix(nx, ny);
    
    if (host_matrixA == NULL || host_matrixB == NULL || host_matrixC == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    //filling matrices with random values
    printf("Fill matrices\n");
    fillMatrixRandom(host_matrixA, nx, ny);
    fillMatrixRandom(host_matrixB, nx, ny);
    fillMatrixRandom(host_matrixC, nx, ny);
    auto start = high_resolution_clock::now();
    for(int x = 0; x < nx; x++)
    {
        for(int y = 0; y < ny; y++)
        {
            host_matrixC[x+y*ny] = host_matrixA[x+y*ny] + host_matrixB[x+y*ny];
        }
    }
    auto stop = high_resolution_clock::now();
    auto durationOnCUDA = duration_cast<microseconds>(stop - start);
    output << durationOnCUDA.count();
}

int main(void)
{
    int filteredNum = 0;
    output.open("output.csv");
    // twelve iterations
    for(int numElements = 0x2; 0x0 == (filteredNum = numElements & 0xFFFFF800) ; numElements = (numElements << 1))
    {
        
        const int nx = numElements;
        const int ny = numElements;

        output << numElements << ",";
        printf("Case elements: %d\n", numElements*numElements);
        if(numElements <= 32) //that will make 1024 threds in single block, which is maximum
        exec_1D1D(ny, nx);
        output << ",";
        exec_1D2D(ny, nx);
        output << ",";
        exec_2D2D(ny, nx);
        output << ",";
        exec_CPU(ny, nx);
        output << std::endl;
        
    }
    output.close();
    // file_1D2D.close();
    // file_2D2D.close();
    // file_CPU.close();

    return 0;
}
