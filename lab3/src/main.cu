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

__global__ void matrixAdd(const float* MatA, const float* MatB, float* MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ids = iy*nx + ix;

	if(i < nx && j < ny)
	{
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

float* allocateFloatMatrix(int nx, int ny)
{
	float* handle = (float*)malloc(nx * ny * sizeof(float));

	return handle;
}

void fillMatriceRandom(float* matrice, int nx, int ny)
{
    for(int j = 0; j < nx; j++)
    {
    	for(int i = 0; i < ny; i++)
    	{
    	    matrice[j * nx + i] = rand()/(float)RAND_MAX;
    	}
    }
}

int main(void)
{
    const int nx = 32;
    const int ny = 32;
    
    const int sizeOfAllocationOnGraphicsCard = nx * ny * sizeof(float);    


    float* host_matrixA = allocateFloatMatrix(nx, ny);
    float* host_matrixB = allocateFloatMatrix(nx, ny);
    float* host_matrixC = allocateFloatMatrix(nx, ny);
    
    //device allocation
    float* dev_matrixA, *dev_matrixB, *dev_matrixC;
    cudaMalloc((void**)&dev_matrixA, sizeOfAllocationOnGraphicsCard);    
    cudaMalloc((void**)&dev_matrixB, sizeOfAllocationOnGraphicsCard);
    cudaMalloc((void**)&dev_matrixC, sizeOfAllocationOnGraphicsCard);
    
    //filling matrices with random values
    printf("Fill matrices\n");
    fillMatriceRandom(host_matrixA, nx, ny);
    fillMatriceRandom(host_matrixB, nx, ny);
    fillMatriceRandom(host_matrixC, nx, ny);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
   
    //copying data on graphics card
    cudaMemcpy(dev_matrixA, host_matrixA, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice);

    cudaMemcpy(dev_matrixB, host_matrixB, sizeOfAllocationOnGraphicsCard, cudaMemcpyHostToDevice);

    // Launch the matrixAdd CUDA Kernel
    dim3 threadsInBlock(nx, ny);
    int threadsPerBlock = nx * ny;
    int blocksPerGrid = 1;
    printf("%d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matrixAdd<<<blocksPerGrid, threadsInBlock>>>(dev_matrixA, dev_matrixB, dev_matrixC, nx, ny);

    cudaMemcpy(host_matrixC, dev_matrixC, sizeOfAllocationOnGraphicsCard, cudaMemcpyDeviceToHost);


    // Verify that the result vector is correct
    for (int j = 0; j < nx; j++)
    {
    	for(int i = 0; i < ny; i++)
    	{
    		printf("%d\t", host_matrixC[j * nx + i])
    	}
    printf("\n")
    }
    // Free host memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_matrixC);

    printf("Done\n");
    return 0;
}

