# Report Lab7

Kamil Gwiżdż & Bartłomiej Mucha

## Device query
The frist exercise was to inspect and compare device query code result from OpenCL and CUDA.
The output from OpenCL C++ and C were the same. There was the basic informations about ours devices. When we looked into the source code we can see that the C++ code was ported to C.
![OpenCL](../lab7/OpenCL.png)
When we look at output from CUDA's DeviceQuery it has much more details about the device capabilities. 
![CUDA](../lab7/CUDA.png)
Comparing this two outputs we can see that there are some diffrents in names. What we have known as *Multiprocessors* in CUDA are named in OpenCL as *Compute Units*. *Max work-group Dims in OpenCL* is *Max dimension size of a thread block in CUDA*.   
## Vector addition example
In the second task, we had to compare OpenCL implementations of vector addition in C and C++ , and then confront the results to CUDA implementation.
First when we compare code in C and C++ we can see that the way of handling OpenCL kernel in C is a little messy. We think that this code could 
be error-prone because a kernel is a simply string so for example finding a bug in string could be more time-consuming than to find a bug in a code with
syntax highlighting. C++ way of handling OpenCL is much more cleaner. The kernel implementation is stored in a diffrent file so in our opinion it is more clear.
We have used already prepared code to made some measurements.But for C++ code we decided to change the way of measuring time because default way did not give precise results.
So we changed the method to chrono high_resolution_clock.

![Compare C and CPP](../lab7/C_CPP_2.png)
![Compare C and CPP 2](../lab7/C_CPP_1.png)

When we look at the chart it shows that C and C++ performance is quite similar. For smaller vectors C code may be slightly faster.

![Compare C, CPP and CUDA](../lab7/C_CPP_CUDA_2.png)
![Compare C,CPP and CUDA 2](../lab7/C_CPP_CUDA_1.png)

After we used CUDA addVector code from NVIDIA CUDA Samples and compare it to C/C++ implementations there were no doubts that CUDA is the fastest one.
It was not a surprise to us because we think that framework for the one type of GPU could be easier to optimize than the framework which is portable.
