# Report Lab6

Kamil Gwiżdż & Bartłomiej Mucha

## SAXPY optimization
“Single-Precision A·X Plus Y”. This is so common operation that its got it own name. SAXPY is a combination of scalar multiplication and vector addition.
Our goal for this task was to optimize this operation to running under 25us. The code from which we started work had some bugs and we were supposed to find and fix them.
After that when we start this code for the first time time to execute kernel was about **21.8ms** on average. Then the large increase in performance for our SAXPY
ensured use of prefetching technique. Time has reduced to **283us!!** So the next step was to use grid stride technique. After several attempts, I decided that the best choice
would be to use 256 threads per block and 4096 blocks per grid. At the end we managed to reduce time to **64us**. This is not below 25us as we assumed at the beginning but when we
compare this result to the result at the beginning we can say that this is huge performance improvement! 

