# Report Lab6

Kamil Gwiżdż & Bartłomiej Mucha

## Streams vector additions
In lab 6 our mission was to push the code to the limit to get the best performance. Our first task was to use our *vector add* code with additions of streams and check if it makes compute faster. But first what are streams and how they work. Simply a stream is a queue of devie work. With streams CUDA operations in diffrent streams may run concurrently and CUDA operations from diffrent streams may be interleaved. So to check how streams affect on execution time we have made many measurements with diffrent data size and with various combinations, with one, two, three and completly without streams.

Execution time for vector length ![n14](https://latex.codecogs.com/gif.latex?%5Clarge%202%5E%7B14%7D)
|                    | Default streams| 1 stream      | 2 streams        |3 streams  |
|--------------------| -------------  |---------------| -----------------|-----------|
|**addVectorInto**   | 9.98us         | 9.95us        | 10.14us          |10.12us    |
|**initWith**        | 8.28us         | 8.34us        | 8.33us           |8.35us     |


Execution time for vector length ![n24](https://latex.codecogs.com/gif.latex?%5Clarge%202%5E%7B24%7D)
|                    | Default streams| 1 stream      | 2 streams        |3 streams  |
|--------------------| -------------  |---------------| -----------------|-----------|
|**addVectorInto**   | 1.87ms         | 1.87ms        | 1.87ms           |1.87ms     |
|**initWith**        | 534.25us       | 536.52us      | 556.45us         |591.56us   |



Execution time for vector length ![n28](https://latex.codecogs.com/gif.latex?%5Clarge%202%5E%7B28%7D)
|                    | Default streams| 1 stream      | 2 streams        |3 streams  |
|--------------------| -------------  |---------------| -----------------|-----------|
|**addVectorInto**   | 2.44s          | 2.6s          | 2.45s            |2.6s       |
|**initWith**        | 936.42ms       | 945.13us      | 1.05s            |1.16s      |

Our first thought was that added streams would increase execution speed but after analyse the results we can see that the performance of function *initWith* slightly decerase. So kernel parallelization can be very useful but for our task it did not provide us performance boost.

## SAXPY optimization
“Single-Precision A·X Plus Y”. This is so common operation that its got it own name. SAXPY is a combination of scalar multiplication and vector addition.
Our goal for this task was to optimize this operation to running under 25us. The code from which we started work had some bugs and we were supposed to find and fix them.
After that when we start this code for the first time time to execute kernel was about **21.8ms** on average. Then the large increase in performance for our SAXPY
ensured use of prefetching technique. Time has reduced to **283us!!** So the next step was to use grid stride technique. After several attempts, I decided that the best choice
would be to use 256 threads per block and 4096 blocks per grid. At the end we managed to reduce time to **64us**. This is not below 25us as we assumed at the beginning but when we
compare this result to the result at the beginning we can say that this is huge performance improvement! 

## Conclusions
Adding streams without thinking about a problem is not the best solution. We should always prepare more then one possible implementation of a problem and then compare result to choose the best way. In our case adding streams did not give any positive results. When we look at our SAXPY problem after few optimizations  we have achieved a huge performance boost. We can see that the use of prefetching technique was of the greatest importance in this case. We also can not ignore the impact of threads per block and number of blocks because when we optimize it our result was about 5 time faster than without optimization. 
