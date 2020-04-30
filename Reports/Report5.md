# Report Lab5

Kamil Gwiżdż & Bartłomiej Mucha

## Unified memory
Unified Memory is a single memory address space accessible from any processor in a system. This allows appliciations to allocate data from code running on CPUs or GPUs. If we want to allocating Unified Memory we can easily calls to ```cudaMallocManaged()``` an allocation function. ....................... Write something about page faults 
..............................

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

Result withoud prefetching
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

We can see that after prefetch all vectors there are no longer any GPU page faults reported and Host to Device transfers is higher.
