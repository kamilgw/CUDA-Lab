#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively
 * and use profiler to check your progress
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 25us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int n, float a, float * x, float * y)
{
    int tid = blockIdx.x * blockDim.x * threadIdx.x;

    if ( tid < N )
        y[tid] = a * x[tid] + y[tid];
}

void initWith(float num, float *a, int size)
{
    for(int i = 0; i < size; ++i)
    {
        a[i] = num;
    }
}

int main()
{
    float *x, *y, *d_x, *d_y;

    int size = N * sizeof (float); // The total number of bytes per vector
    int deviceId;
    cudaGetDevice(&deviceId);

    x = (float*)malloc(size);
    y = (float*)malloc(size);

    cudaMalloc(&d_x, size); 
    cudaMalloc(&d_y, size);

    // Initialize memory
    initWith(2., x, N);
    initWith(1., y, N);


    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int number_of_blocks = 4092;

    saxpy <<< number_of_blocks, threads_per_block >>> ( N, 2.0, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t asyncErr;
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, y[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, y[i]);
    printf ("\n");
}
