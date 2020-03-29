# Report Lab2

Kamil Gwiżdż & Bartłomiej Mucha

## Processing grid improvement
Our first task was to improve the code which creates the **processing grid**. As we can see the code is inefficiently 
because the processing grid is calculated depending on the amount of data to process. This way of describing grid is quite correct
but if we know available resources offered by the GPUs, it means how many streaming multiprocessors (SM) are there and how many cuda
cores are available for each SM, we can reach the maximum performance of our GPU. 

We can use the code from the first class to get this information.
```cuda
cudaSetDevice(dev);
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
int countMP = deviceProp.multiProcessorCount;
int cudaCoresPerMP = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
```

Having number of streaming multiprocessors and cuda cores for each SM we can create a grid with *countMP* blocks and for 
each block *cudaCoresPerMP* number of threads to execute our kernel function.

## Limiting factors
Our next task was to found the limitations of *addVec* code. We checked how many elements can be processed using this code.
So we started increasing the numElements in vector until our program crash. We found that program working with 10e8 but crashed
with 10e9. In each array we are using float number so for each number it is 4 bytes in memory. We can simply calculate the sie of used memory.

```math #sum
size = 3 * 10e8 * 4 bytes
```
