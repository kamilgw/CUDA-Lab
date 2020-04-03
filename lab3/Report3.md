# Report Lab3

Kamil Gwiżdż & Bartłomiej Mucha

## Grid layout vs performance
The topic of the current task is checking of how does the grid layout impacts the overall performance. The example problem that was put under the investigation is matrix addition algorithm.
```math #sum
C = A + B
```
All cells were calculated individually in different threads, however the layout of blocks and grid was different as following approaches of building a grid layout were made:
 1. All threads in one dimension are placed in one block.
 2. Threads are placed in multiple one dimensional block in two dimensional grid.
 3. Threads are placed in multiple two dimensional block in two dimensional grid.
 4. For comaparision same task was executed on CPU in single thread.

The first approach quite quickly showed error about invalid CUDA kernel arguments, as maximum number  of threads per single block is 1024. Therefore curve "1D1D CUDA" on the charts ends on the case of the size of a matrices is 32 (1024 elements).

```cuda
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

The second and the third approach showed almost same performance. Worthy to mention is that the third approach has at the begining slighlty better performance. However what is important about it, it is the most human-friendly approach among all described, beacuse the layout of threads can be drawn the same way the matrix would be. Grid layout and matrix' cells covers in that case.

In the end the linear calculation on single CPU thread appeard to be more efficient for matrices which size is quite small, on the hardware we were using the size was about 37.

## Time comparison
After that we started to measure the execution time of our code. We added some code which add vectors using CPU. We made a few measurements for various input vector size.   
![Compare CPU and CUDA](Chart3.png)
On the first plot we can see that execution time on CUDA is constant in contrast to CPU where time increase linearly. But if we look closer on the cases where the vector size is smaller than 10000.
![Compare CPU and CUDA](Chart4.png)
We can see that prepare GPU to work costs about 10ms so CUDA cores are faster than CPU only if we are adding vectors larger than ~5000 elements. So if we want to add vectors that have a few elements a better choice might be to use CPU.
