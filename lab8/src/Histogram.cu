/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 * Histogram
*/

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
#include <assert.h>
#include <limits.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std::chrono;

// # operator doesn't work :(
#define MAP "32"
#define BLOCK_SIZE 1024
#define VALIDATE 0
#define MANAGED 1
#define DEVICE_HOST 0

#define SPEC ""
#if MANAGED + DEVICE_HOST == 1
    #if MANAGED
        #undef SPEC
        #define SPEC "managed_only"
    #else
        #undef SPEC
        #define SPEC "dev_host_only"
    #endif
#endif

#define CPU "CPU"
#define GPU "GPU"
#define ENABLE_CPU 1
#define ENABLE_GPU 1
#define nBins 2048
#define STRIDE 512
#define MAX_INPUT_VALUE 4096
//#define VEC_SIZE 1073741824 // 0x40000000
//#define VEC_SIZE 268435456 // 0x10000000
//#define VEC_SIZE 268435455 // 0xFFFFFFF
//#define VEC_SIZE 33554431 // 0x1FFFFFF
//#define VEC_SIZE 1048575 // 0xFFFFF
#define VEC_SIZE 16777215 // 0xFFFFFF

// GPU Histogrm generation
__global__ void histogrammizeVector(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int strideX = blockDim.x*gridDim.x;
    const float BinSize = maxInputValue/(float)OutputSize;

    if(colIdx == 0) printf("GPU BinSize: %f\n", BinSize);
    if(colIdx == 0) printf("GPU strideX: %d\n", strideX);
    
    for(int i = colIdx; i < InputSize; i+=strideX)
    {
        int binNr = (int)floorf(Input[i]/BinSize);
        if(i == 0) printf("GPU binNr: %d\n", binNr);
        if( Output[binNr] <= USHRT_MAX ) // ommit if max value of bin is already reached
        {
            (void)atomicAdd( &Output[binNr], 1); //increment
        }
    }
    
}

// GPU Histogrm generation
__global__ void histogrammizeVector2(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunkIdx = InputSize/blockDim.x*gridDim.x;
    const float BinSize = maxInputValue/(float)OutputSize;

    if(colIdx == 0) printf("GPU BinSize: %f\n", BinSize);
    
    for(int i = colIdx*chunkIdx; i < (colIdx+1)*chunkIdx; i++)
    {
        if(i > InputSize) return;
        int binNr = (int)floorf(Input[i]/BinSize);
        if(i == 0) printf("GPU binNr: %d\n", binNr);
        if( Output[binNr] <= USHRT_MAX ) // ommit if max value of bin is already reached
        {
            (void)atomicAdd( &Output[binNr], 1); //increment
        }
    }
    
}

// GPU Histogrm generation
__global__ void histogrammizeVector_kernelPerBin(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float BinSize = maxInputValue/(float)OutputSize;
    int count = 0;
    if(colIdx == 0) printf("GPU BinSize: %f\n", BinSize);
    
    for(int i = 0; i < InputSize; i++)
    {
        if(floorf(Input[i]) >= colIdx*BinSize && floorf(Input[i]) < (colIdx+1)*BinSize)
            count++;
    }

    (void)atomicAdd( &Output[colIdx], count & USHRT_MAX); //add
}

// CPU Histogram generation
void histogrammizeVectorCPU(const float * const Input, int* Output, const float maxInputValue, const int InputSize, const int OutputSize){

    const float BinSize = maxInputValue/(float)OutputSize;
    printf("CPU BinSize: %f\n", BinSize);
    for(int i = 0; i < InputSize; i++)
    {
        
        int binNr = static_cast<int>(Input[i]/BinSize); // assign value to corresponding bin
        
        if( Output[binNr] < USHRT_MAX ) // ommit if max value of bin already reached
        {
            Output[binNr]++;
        }
    }
}

void writeHistogramToFile(std::ofstream & save, int* const Output, const float maxInputValue, const int OutputSize)
{
    const float BinSize = maxInputValue/(float)OutputSize;

    save << "Size of bin" <<";" << "Value" <<";" <<std::endl;
    int i = 0;
    for(; i < OutputSize; i++)
        save << BinSize*i <<";" << Output[i] <<";" <<std::endl;
    save << BinSize*(i+1) <<";" <<std::endl; //for gnuplot
}

void writeHistogramToFile2(std::ofstream & save, float* Input, const float maxInputValue, const int InputSize)
{
    int i = 0;
    for(; i < InputSize; i++)
        save << Input[i] <<";" <<std::endl;
}

template<class T>
void initializeVector(T *A, const unsigned long numberOfAllElements){
    for(int i = 0; i < numberOfAllElements; i++){
        A[i] = rand()%MAX_INPUT_VALUE;
    }
}

inline cudaError_t checkCuda(cudaError_t result, int line = -1) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << " " << line <<std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

template<class T>
void initWith(const T num, T * const a, const int N)
{
    for(int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}


int main() {
    float * host_vectorInput, *dev_vectorInput;
    int * outputCPU, * host_outputGPU, *dev_outputGPU, *host_outputGPUgpk, *dev_outputGPUgpk, *host_outputGPU2, *dev_outputGPU2;

    const char * const filename_gpu = "out" MAP GPU ".csv";
    const char * const filename_cpu = "out" MAP CPU ".csv";
    const char * const filename_gpu_bpk = "out" MAP GPU "_bpk.csv";
    const char * const filename_gpu2 = "out" MAP GPU "2.csv";
    int MaxInputValue = MAX_INPUT_VALUE;
    int time;
    std::ofstream save;

    int numberOfElements = VEC_SIZE;
    size_t size = numberOfElements * sizeof(float);
    size_t sizeHist = nBins * sizeof(int);

    dim3 threads_per_block (BLOCK_SIZE, 1, 1);
    dim3 number_of_blocks ((VEC_SIZE / (BLOCK_SIZE * nBins))-1, 1, 1);

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point stop;

    std::cout<<"start"<<std::endl;

    std::cout <<"Alloc vectors"<<std::endl;

    outputCPU = static_cast<int*>(malloc(sizeHist));
    host_outputGPU = static_cast<int*>(malloc(sizeHist));
    host_vectorInput = static_cast<float*>(malloc(size));
    host_outputGPU2 = static_cast<int*>(malloc(sizeHist));
    host_outputGPUgpk = static_cast<int*>(malloc(sizeHist));
    checkCuda(cudaMalloc((void**)&dev_outputGPU, sizeHist),__LINE__);
    checkCuda(cudaMalloc((void**)&dev_vectorInput, size), __LINE__);
    checkCuda(cudaMalloc((void**)&dev_outputGPU2, sizeHist), __LINE__);
    checkCuda(cudaMalloc((void**)&dev_outputGPUgpk, sizeHist), __LINE__);
    
    // Initialize memory
    std::cout <<"Init vectors"<<std::endl;
    initWith((int)0, outputCPU, nBins);
    initWith((int)0, host_outputGPU, nBins);
    initWith((int)0, host_outputGPU2, nBins);
    initWith((int)0, host_outputGPUgpk, nBins);
    initializeVector(host_vectorInput, numberOfElements);

    checkCuda(cudaMemcpy(dev_outputGPU, host_outputGPU, sizeHist, cudaMemcpyHostToDevice), __LINE__);
    checkCuda(cudaMemcpy(dev_vectorInput, host_vectorInput, size, cudaMemcpyHostToDevice), __LINE__);
    checkCuda(cudaMemcpy(dev_outputGPU2, host_outputGPU2, sizeHist, cudaMemcpyHostToDevice), __LINE__);
    checkCuda(cudaMemcpy(dev_outputGPUgpk, host_outputGPUgpk, sizeHist, cudaMemcpyHostToDevice), __LINE__);

    #if ENABLE_GPU
    {
        std::cout <<"GPU"<<std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout <<"MaxInputValue: "<< MaxInputValue << ", numberOfElements: " << numberOfElements << ", nBins: " << nBins <<std::endl;
        histogrammizeVector<<<number_of_blocks, threads_per_block>>>(
                    dev_vectorInput, 
                    dev_outputGPU, 
                    MaxInputValue, 
                    numberOfElements, 
                    nBins);
        std::cout <<"numberOfElements/MaxInputValue: "<< numberOfElements/MaxInputValue <<std::endl;
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        auto elapsed_time = duration_cast<microseconds>(stop - start);
        time = elapsed_time.count();
        std::cout << "Time: " << time << std::endl;
        checkCuda(cudaMemcpy(host_outputGPU, dev_outputGPU, sizeHist, cudaMemcpyDeviceToHost), __LINE__);
        checkCuda(cudaFree(dev_outputGPU), __LINE__);
        #if VALIDATE
        try{
            // validateMatrix(host_result_CPU, result_stride, numberOfElements);
        }
        catch(const char* const e)
        {
            std::cout<<e<<std::endl;
        }
        #endif
        std::cout <<"\twriting to file..."<<std::endl;
        save.open(filename_gpu);
        writeHistogramToFile(save, host_outputGPU, MaxInputValue, nBins);
        save.close();
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::cout <<"GPU"<<std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout <<"MaxInputValue: "<< MaxInputValue << ", numberOfElements: " << numberOfElements << ", nBins: " << nBins <<std::endl;
        histogrammizeVector_kernelPerBin<<<nBins/threads_per_block.x, threads_per_block>>>(
                    dev_vectorInput, 
                    dev_outputGPUgpk, 
                    MaxInputValue, 
                    numberOfElements, 
                    nBins);
        std::cout <<"numberOfElements/MaxInputValue: "<< numberOfElements/MaxInputValue <<std::endl;
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        elapsed_time = duration_cast<microseconds>(stop - start);
        time = elapsed_time.count();
        std::cout << "Time: " << time << std::endl;
        checkCuda(cudaMemcpy(host_outputGPUgpk, dev_outputGPUgpk, sizeHist, cudaMemcpyDeviceToHost), __LINE__);
        checkCuda(cudaFree(dev_outputGPUgpk), __LINE__);
        #if VALIDATE
        try{
            // validateMatrix(host_result_CPU, result_stride, numberOfElements);
        }
        catch(const char* const e)
        {
            std::cout<<e<<std::endl;
        }
        #endif
        std::cout <<"\twriting to file..."<<std::endl;
        save.open(filename_gpu_bpk);
        writeHistogramToFile(save, host_outputGPUgpk, MaxInputValue, nBins);
        save.close();
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::cout <<"GPU"<<std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout <<"MaxInputValue: "<< MaxInputValue << ", numberOfElements: " << numberOfElements << ", nBins: " << nBins <<std::endl;
        histogrammizeVector2<<<number_of_blocks, threads_per_block>>>(
                    dev_vectorInput, 
                    dev_outputGPU2, 
                    MaxInputValue, 
                    numberOfElements, 
                    nBins);
        std::cout <<"numberOfElements/MaxInputValue: "<< numberOfElements/MaxInputValue <<std::endl;
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        elapsed_time = duration_cast<microseconds>(stop - start);
        time = elapsed_time.count();
        std::cout << "Time: " << time << std::endl;
        checkCuda(cudaMemcpy(host_outputGPU2, dev_outputGPU2, sizeHist, cudaMemcpyDeviceToHost), __LINE__);
        checkCuda(cudaFree(dev_outputGPU2), __LINE__);
        #if VALIDATE
        try{
            // validateMatrix(host_result_CPU, result_stride, numberOfElements);
        }
        catch(const char* const e)
        {
            std::cout<<e<<std::endl;
        }
        #endif
        std::cout <<"\twriting to file..."<<std::endl;
        save.open(filename_gpu2);
        writeHistogramToFile(save, host_outputGPU2, MaxInputValue, nBins);
        save.close();
    }
    #endif

    #if ENABLE_CPU
    {
        std::cout <<"CPU"<<std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout <<"MaxInputValue: "<< MaxInputValue << ", numberOfElements: " << numberOfElements << ", nBins: " << nBins <<std::endl;
        histogrammizeVectorCPU(
                    host_vectorInput, 
                    outputCPU, 
                    MaxInputValue, 
                    numberOfElements, 
                    nBins);
        stop = std::chrono::high_resolution_clock::now();
        auto elapsed_time = duration_cast<microseconds>(stop - start);
        time = elapsed_time.count();
        std::cout << "Time: " << time << std::endl;
        #if VALIDATE
        try{
            // validateMatrix(host_result_CPU, result_stride, numberOfElements);
        }
        catch(const char* const e)
        {
            std::cout<<e<<std::endl;
        }
        #endif
        std::cout <<"\twriting to file..."<<std::endl;
        save.open(filename_cpu);
        writeHistogramToFile(save, outputCPU, MaxInputValue, nBins);
        save.close();
    }
    #endif

    checkCuda(cudaFree(dev_vectorInput), __LINE__);
       
    free(outputCPU);
    free(host_outputGPU);
    free(host_outputGPU2);
    free(host_outputGPUgpk);
    free(host_vectorInput);

}
