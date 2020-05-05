/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 * Matrix multiplication C=A*B
*/

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
#include <assert.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std::chrono;

__global__ void matrixMultiplication(const double *A, const double *B, double *C, int size) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(rowIdx < size && colIdx < size){
        double result = 0.;
        for(int i = 0; i < size; i++){
            result += A[rowIdx * size + i] * B[i * size + colIdx];
        }
        C[rowIdx * size + colIdx] = result;
    }
}


// Check if it works
__global__ void strideMatrixMultiplication(const double * const A, const double* const B, double* const C, int size){
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    for(int i = colIdx; i < size; i += strideX){
        for(int j = rowIdx; j < size; j += strideY){
            double result = 0.;
            for(int k = 0; k < size; k++){
                result += A[j * size + k] * B[k * size + i];
            }
            C[j * size + i] = result;
        }
    }
}

void cpuMatrixMultiplication(const double * const A, const double* const B, double* const C, int size){
    for(int i = 0; i < size; i ++){
        for(int j = 0; j < size; j ++){
            double result = 0.;
            for(int k = 0; k < size; k++){
                result += A[j * size + k] * B[k * size + i];
            }
            C[j * size + i] = result;
        }
    }
}

void validateMatrix(const double * const A, const double * const B, const int size){
    double sub;

    for(int i = 0; i < size; i++)
    {
        sub = A[i]-B[i];
        if( sub > 1.0e-06 || sub < -1.0e-06)
            throw "Matrix mismatch";
    }
}

void initializeMatrix(double *A, unsigned long numberOfAllElements){
    for(int i = 0; i < numberOfAllElements; i++){
        A[i] = rand()/(double)RAND_MAX;
    }
}

// void allocateHost(double** A, const size_t size) {
//     *A = (double*) malloc(size);

//     if(A == nullptr) {
//         std::cout << "Can not allocate host memory" << std::endl;
//         exit(0);
//     }
// }

// void allocateDevice(double * A, size_t size) {
//     checkCuda(cudaMalloc((void**)A, size));
// }

// void freeDevice(double *A) {
//     checkCuda(cudaFree(A));
// }

void printMatrix(double * A, unsigned long N, unsigned long M) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++)
            std::cout << A[i * N + j] << " ";
        std::cout << std::endl;
    }
}

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
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

#define DIM1 32
#define DIM2 32
// # operator doesn't work :(
#define sDIM1 "32"
#define sDIM2 "32"
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


int main() {
    double *host_A, *host_B, *host_result_CPU, *host_result_stride, *host_result;
    double *dev_A, *dev_B, *dev_result, *dev_result_stride;
    double *A, *B, *result, *result_stride;
    int deviceId;
    const char * const filename = "out" sDIM1 "_" sDIM2 "_" SPEC ".csv";
    std::ofstream save;
    save.open(filename);

    int numberOfElementsInDim;
    int numberOfElements;
    size_t size;

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point stop;

    int startnum = 100;
    int jump = 100;
    int numberOfResult = 70;
    int numberOfIteration = 1;
    bool check = false;
    double time, timeStride;
    double timeManaged, timeStrideManaged;

    std::cout<<"start"<<std::endl;

    numberOfElementsInDim = startnum;
    numberOfElements = numberOfElementsInDim * numberOfElementsInDim;
    size = numberOfElements * sizeof(double);

    std::cout <<"classic_2D"<<std::endl;
    std::cout << std::setw(20)  << "numbOfElements" << std::setw(20) <<"numOfElementsInDim" << std::setw(20) <<"time" << std::setw(20) <<"time stride" << std::setw(20) <<"time managed" << std::setw(20) <<"time stride managed" << std::endl;
    save << "number of elements;time;time stride;time managed;time stride managed" << std::endl;
    for(int i = 0;i < numberOfResult; i++) {
        time = 0.;
        timeStride = 0.;
        timeManaged = 0.;
        timeStrideManaged = 0.;
        //for(int k = 0; k < numberOfIteration; k++){
            

            host_A = static_cast<double*>(malloc(size));
            host_B = static_cast<double*>(malloc(size));
            host_result = static_cast<double*>(malloc(size));
            host_result_stride = static_cast<double*>(malloc(size));
            host_result_CPU = static_cast<double*>(malloc(size));
            checkCuda(cudaGetDevice(&deviceId));

            checkCuda(cudaMallocManaged(&A, size));
            checkCuda(cudaMallocManaged(&B, size));
            checkCuda(cudaMallocManaged(&result, size));
            checkCuda(cudaMallocManaged(&result_stride, size));
            // Initialize memory
            initWith(0., result, numberOfElements);
        
            initializeMatrix(host_A, numberOfElements);
            initializeMatrix(host_B, numberOfElements);
            checkCuda(cudaMemPrefetchAsync(A, size, deviceId));
            checkCuda(cudaMemPrefetchAsync(B, size, deviceId));
            checkCuda(cudaMemPrefetchAsync(result, size, deviceId));

            checkCuda(cudaMalloc((void**)&dev_A, size));
            checkCuda(cudaMalloc((void**)&dev_B, size));
            checkCuda(cudaMalloc((void**)&dev_result, size));
            checkCuda(cudaMalloc((void**)&dev_result_stride, size));

            checkCuda(cudaMemcpy(dev_A, host_A, size, cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(dev_B, host_B, size, cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(A, host_A, size, cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(B, host_B, size, cudaMemcpyHostToDevice));
            #if VALIDATE
            cpuMatrixMultiplication(host_A, host_B, host_result_CPU, numberOfElementsInDim);
            #endif
            dim3 threads_per_block (DIM1, DIM2, 1);
            dim3 number_of_blocks ((numberOfElementsInDim / threads_per_block.x) + 1, (numberOfElementsInDim / threads_per_block.y) + 1, 1);

            #if DEVICE_HOST
            {
                start = std::chrono::high_resolution_clock::now();
                matrixMultiplication<<<number_of_blocks, threads_per_block>>>(dev_A, dev_B, dev_result, numberOfElementsInDim);
                cudaDeviceSynchronize();
                stop = std::chrono::high_resolution_clock::now();
                auto elapsed_time = duration_cast<microseconds>(stop - start);
                time += elapsed_time.count();
                checkCuda(cudaMemcpy( host_result, dev_result, size, cudaMemcpyDeviceToHost));
                #if VALIDATE
                try{
                    validateMatrix(host_result_CPU, host_result, numberOfElements);
                }
                catch(const char* const e)
                {
                    std::cout<<e<<std::endl;
                }
                #endif
            }
            #endif
            checkCuda(cudaFree(dev_result));

            #if DEVICE_HOST
            {
                start = std::chrono::high_resolution_clock::now();
                strideMatrixMultiplication<<<number_of_blocks, threads_per_block>>>(dev_A, dev_B, dev_result_stride, numberOfElementsInDim);
                cudaDeviceSynchronize();
                stop = std::chrono::high_resolution_clock::now();
                auto elapsed_time = duration_cast<microseconds>(stop - start);
                timeStride += elapsed_time.count();
                checkCuda(cudaMemcpy( host_result_stride, dev_result_stride, size, cudaMemcpyDeviceToHost));
                #if VALIDATE
                try{
                    validateMatrix(host_result_CPU, host_result_stride, numberOfElements);
                }
                catch(const char* const e)
                {
                    std::cout<<e<<std::endl;
                }
                #endif
            }
            #endif
            checkCuda(cudaFree(dev_result_stride));

            #if MANAGED
            {
                start = std::chrono::high_resolution_clock::now();
                matrixMultiplication<<<number_of_blocks, threads_per_block>>>(A, B, result, numberOfElementsInDim);
                cudaDeviceSynchronize();
                stop = std::chrono::high_resolution_clock::now();
                auto elapsed_time = duration_cast<microseconds>(stop - start);
                timeManaged += elapsed_time.count();
                #if VALIDATE
                try{
                    validateMatrix(host_result_CPU, result, numberOfElements);
                }
                catch(const char* const e)
                {
                    std::cout<<e<<std::endl;
                }
                #endif
            }
            #endif
            checkCuda(cudaFree(result));

            #if MANAGED
            {
                start = std::chrono::high_resolution_clock::now();
                strideMatrixMultiplication<<<number_of_blocks, threads_per_block>>>(A, B, result_stride, numberOfElementsInDim);
                cudaDeviceSynchronize();
                stop = std::chrono::high_resolution_clock::now();
                auto elapsed_time = duration_cast<microseconds>(stop - start);
                timeStrideManaged += elapsed_time.count();
                #if VALIDATE
                try{
                    validateMatrix(host_result_CPU, result_stride, numberOfElements);
                }
                catch(const char* const e)
                {
                    std::cout<<e<<std::endl;
                }
                #endif
            }
            #endif
            checkCuda(cudaFree(result_stride));

            checkCuda(cudaFree(dev_A));
            checkCuda(cudaFree(dev_B));
            checkCuda(cudaFree(A));
            checkCuda(cudaFree(B));
            free(host_A);
            free(host_B);
            free(host_result);
            free(host_result_stride);
            free(host_result_CPU);
        //}

        save << numberOfElementsInDim <<";" << time/(float)numberOfIteration << ";" << timeStride/(float)numberOfIteration 
        << ";" << timeManaged/(float)numberOfIteration << ";" << timeStrideManaged/(float)numberOfIteration << std::endl;
        std::cout << std::setw(20) << numberOfElements << std::setw(20)<< numberOfElementsInDim 
                  << std::setw(20)<< time/(float)numberOfIteration << std::setw(20) << timeStride/(float)numberOfIteration 
                  << std::setw(20)<< timeManaged/(float)numberOfIteration << std::setw(20) << timeStrideManaged/(float)numberOfIteration << std::endl;

        numberOfElementsInDim += jump;
        numberOfElements = numberOfElementsInDim * numberOfElementsInDim;
        size = numberOfElements * sizeof(double);
    }

    save.close();


    numberOfElementsInDim = startnum;
    numberOfElements = numberOfElementsInDim * numberOfElementsInDim;
    size = numberOfElements * sizeof(double);

}
