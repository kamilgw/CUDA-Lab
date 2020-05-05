# Report Lab3

Kamil Gwiżdż & Bartłomiej Mucha

## Grid layout vs performance
```math #sum
C = A * B
```
```cuda
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```


## Time comparison

![Compare CPU and CUDA](../lab4/AllLayoutsAverage.PNG)

![Compare CPU and CUDA](../lab4/3x7layout.PNG)
![Compare CPU and CUDA](../lab4/10x10layout.PNG)
![Compare CPU and CUDA](../lab4/32x32layout.PNG)

## Conclusion


