# Report Lab8

Kamil Gwiżdż & Bartłomiej Mucha


_WIP_
## Setup

There are two setups of Input data, let's call them A and B set. Set A consists of 134215680 (127M) elements and B set consists of 1048575 (1M) elements. Values are distrubited quite evenly as all values come from the pseudo-random number engine with range of 0 to 4048. The grid has always layout 1D1D,  where there are 1024 threads per block. This 1D block was proven up until now through previous reports it is the most efficent way. In order to make things more interesting yhe input is has always more elements than all threads in cuda kernel. As many therads may need to access the same element of histogram ouput array all incerment operations are made in atomic call.
```cuda
unsigned int atomicInc(unsigned int* address, unsigned int val);
```

## CPU single thread approach

```cuda
void histogrammizeVectorCPU(const float * const Input, int* Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const float BinSize = maxInputValue/(float)OutputSize;
    for(int i = 0; i < InputSize; i++)
    {
        int binNr = static_cast<int>(Input[i]/BinSize); // assign value to corresponding bin
        if( Output[binNr] < USHRT_MAX ) // ommit if max value of bin already reached
        {
            Output[binNr]++;
        }
    }
}
```

For data sets A execution time is about ????? ns and for the B set ????? ns. We will use this data to compare with parallel approaches.

## Parallel multi-thread approach
### Serial loop calling cuda kernel for each piece of input data

```cuda
__global__ void histogrammizeVector3(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize, const int offset){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float BinSize = maxInputValue/(float)OutputSize;

    if(colIdx == 0) printf("GPU BinSize: %f\n", BinSize);
    
    if(offset+colIdx >= InputSize) return;
    int binNr = (int)floorf(Input[offset+colIdx]/BinSize);
    if(colIdx == 0) printf("GPU binNr: %d\n", binNr);
    (void)atomicInc( (unsigned int*)&Output[binNr], USHRT_MAX+1); //increment
}
```

The input data is split to chunks which are process one by one by on cuda device.
Execution time:
* ????ns
* ????ns

### Serial loop through all input array elements inside cuda kernel
This approach is crealy not a way to use parallel computing. However it doesn't require atomic function to preform calculation.

```cuda
__global__ void histogrammizeVector_kernelPerBin(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float BinSize = maxInputValue/(float)OutputSize;
    int count = 0;
    
    for(int i = 0; i < InputSize; i++)
    {
        if(floorf(Input[i]) >= colIdx*BinSize && floorf(Input[i]) < (colIdx+1)*BinSize)
            count++;
    }

    &Output[colIdx] = count & USHRT_MAX;
}
```
Execution time:
* ????ns
* ????ns

### Input data is proccessed in strides in cuda kernel as serial subarrays
Each thread process a certain amount of consecutive elements from input data. 

```cuda
__global__ void histogrammizeVector2(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunkIdx = InputSize/blockDim.x*gridDim.x;
    const float BinSize = maxInputValue/(float)OutputSize;

    for(int i = colIdx*chunkIdx; i < (colIdx+1)*chunkIdx; i++)
    {
        int binNr = (int)floorf(Input[i]/BinSize);
        (void)atomicInc( (unsigned int*)&Output[binNr], USHRT_MAX+1); //increment
    }
}
```

There is flaw in this method which at some points currupts data on cuda device so it is no longer accesible from host device. Supposedly that is consequence of overusing atomic operation which should be use sporadically in critical section of the code. As every atomic call has to be stacked to be exectued when resource is not in use, there is chance of overflow or triggering some safety mechanism inside cuda device which blocks or frees memory in a silent as error is risen on copying device array to a host one. That's why it only handles a B data set.

Execution time:
* ????ns

### Input data is proccessed in strides in cuda kernel
Each thread process a certain elements from input data separted by a stride length. 

```cuda
__global__ void histogrammizeVector(const float * const Input, int * Output, const float maxInputValue, const int InputSize, const int OutputSize){
    const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int strideX = blockDim.x*gridDim.x;
    const float BinSize = maxInputValue/(float)OutputSize;
    
    for(int i = colIdx; i < InputSize; i+=strideX)
    {
        int binNr = (int)floorf(Input[i]/BinSize);
        (void)atomicInc( (unsigned int*)&Output[binNr], USHRT_MAX+1); //increment
    }
}
```
