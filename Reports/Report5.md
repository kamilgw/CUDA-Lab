# Report Lab5

Kamil Gwiżdż & Bartłomiej Mucha

## Blind optimisation
Firstly on our lab we try to optimize the simple program by guessing. We have tried various configuration of PG to choose the best one.

## Unified memory
Unified Memory is a single memory address space accessible from any processor in a system. This allows appliciations to allocate data from code running on CPUs or GPUs. If we want to allocating Unified Memory we can easily calls to ```cudaMallocManaged()``` an allocation function. Page fault is an execution fault, they occur when there is a need to increase the amount of memory available in program.

**CPU only**
```bash
==573== Unified Memory profiling result:
Total CPU Page faults: 384
```

**GPU only**
```bash
==1483== Unified Memory profiling result:
Device "GeForce RTX 2060 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     574         -         -         -           -  26.55216ms  Gpu page fault groups
```

**CPU->GPU**
```bash
==2034== Unified Memory profiling result:
Device "GeForce RTX 2060 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    6301  20.802KB  4.0000KB  128.00KB  128.0000MB  19.99347ms  Host To Device
     578         -         -         -           -  74.12842ms  Gpu page fault groups
Total CPU Page faults: 384
```

**GPU->CPU**
```bash
==5771== Unified Memory profiling result:
Device "GeForce RTX 2060 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     768  170.67KB  4.0000KB  0.9961MB  128.0000MB  11.04486ms  Device To Host
     574         -         -         -           -  28.32451ms  Gpu page fault groups
Total CPU Page faults: 384
```

After made all measurements we can see that when CPU function was invoked first and then the GPU there were much more page faults than inversely. Also when we compare time of kernel execution it is clearly to see that CPU-GPU is the slowest. When we look to GPU->CPU and only GPU, kernel execution is bit faster on GPU because operating system have to manage the memory only on GPU.
## Prefetching technique

This all about move the data to the GPU after initializiang it. We can do it on CUDA with ```cudaMemPrefetchAsync()```.
To understand how prefetching data can impact on our code we took measurements for various situations:
- Initialise all data structure on GPU onlny
- Prefetch just one vector
- Prefetch two
- Prefetch all vectors

| GPU only      | Prefetch one vec| Prefetch two vec  |Prefetch all|
| ------------- |---------------| -----------------|-----------|
| 365.59ms      | 209.84ms        | 84.44ms           |1.63ms      |

Result without prefetching
```bash
==24217== Unified Memory profiling result:
Device "GeForce RTX 2060 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   10111  38.890KB  4.0000KB  524.00KB  384.0000MB  48.94144ms  Host To Device
     768  170.67KB  4.0000KB  0.9961MB  128.0000MB  11.03334ms  Device To Host
    1416         -         -         -           -  240.2856ms  Gpu page fault groups
Total CPU Page faults: 1536
```

Result with prefetch all vectors
```bash
==9381== Unified Memory profiling result:
Device "GeForce RTX 2060 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     192  2.0000MB  2.0000MB  2.0000MB  384.0000MB  35.78422ms  Host To Device
     768  170.67KB  4.0000KB  0.9961MB  128.0000MB  10.96378ms  Device To Host
Total CPU Page faults: 1536
```

We can see that after prefetch all vectors there are no longer any GPU page faults reported and Host to Device transfers is higher. But when we look at the time of execution it can be impressive how fast is computation prefetching all vectors. It took only 1.6ms! 
