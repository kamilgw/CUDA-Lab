#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>


// cuda thread synchronization
__global__ void
naive_sum_reduction(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // thread synchronous reduction
        if ( (idx_x % (stride * 2)) == 0 )
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}
// cuda thread synchronization
__global__ void
reduction1(float* d_out, float* d_in, unsigned int size)
{

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[tid] = (i < size) ? d_in[i] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        // thread synchronous reduction
        if (index < blockDim.x)
            s_data[index] += s_data[index + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

// cuda thread synchronization
__global__ void
reduction2(float* d_out, float* d_in, unsigned int size)
{

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[tid] = (i < size) ? d_in[i] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

// cuda thread synchronization
__global__ void
reduction3(float* d_out, float* d_in, unsigned int size)
{

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[tid] = (i < size) ? d_in[i] + d_in[i + blockDim.x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void
reduction4(float* d_out, float* d_in, unsigned int size)
{

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[tid] = (i < size) ? d_in[i] + d_in[i + blockDim.x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(s_data, tid);

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}
void reduction(float *d_out, float *d_in, int n_threads, int size)
{   
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    while(size > 1)
    {
        int n_blocks = (size + n_threads - 1) / n_threads;
        reduction_kernel<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(d_out, d_out, size);
        size = n_blocks;
    } 
}

void run_benchmark(void (*reduce)(float*, float*, int, int), 
              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    reduce(d_outPtr, d_inPtr, num_threads, size);

    for (int i = 0; i < test_iter; i++) {
        reduce(d_outPtr, d_inPtr, num_threads, size);
    }

    // getting elapsed time
    cudaDeviceSynchronize();

}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = 1.;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}

int main(int argc, char *argv[])
{
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1024*65535;

    float result_host, result_gpu;

    srand(2019);

    // Allocate memory
    h_inPtr = (float*)malloc(size * sizeof(float));
    
    // Data initialization with random values
    init_input(h_inPtr, size);

    // Prepare GPU resource
    cudaMalloc((void**)& d_inPtr, size * sizeof(float));
    cudaMalloc((void**)&d_outPtr, size * sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

    // Get reduction result from GPU
    run_benchmark(reduction, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    // Get all sum from CPU
    result_host = get_cpu_result(h_inPtr, size);
    printf("host: %f, device %f\n", result_host, result_gpu);
    
    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}



