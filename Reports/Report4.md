# Report Lab4

Kamil Gwiżdż & Bartłomiej Mucha

## Grid layout vs performance
```math #sum
C = A * B
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
```cuda
32x32
 GPU activities:   48.90%  367.003s       140  2.62145s  103.42us  10.3127s  matrixMultiplication(double const *, double const *, double*, int)
                   48.88%  366.852s       140  2.62037s  103.84us  10.2792s  strideMatrixMultiplication(double const *, double const *, double*, int)
```
```cuda
10x10
 GPU activities:   49.04%  430.166s       140  3.07262s  99.772us  12.0150s  matrixMultiplication(double const *, double const *, double*, int)
                   49.03%  430.053s       140  3.07181s  101.76us  11.9901s  strideMatrixMultiplication(double const *, double const *, double*, int)
```
```cuda
3x7
 GPU activities:   49.58%  993.457s       140  7.09612s  188.25us  27.6538s  strideMatrixMultiplication(double const *, double const *, double*, int)
                   49.54%  992.781s       140  7.09129s  188.18us  27.6171s  matrixMultiplication(double const *, double const *, double*, int)
```
```cuda
32x32 device <-> host memory management
 GPU activities:   48.06%  182.824s        70  2.61177s  104.12us  10.2201s  matrixMultiplication(double const *, double const *, double*, int)
                   47.99%  182.542s        70  2.60775s  103.84us  10.1987s  strideMatrixMultiplication(double const *, double const *, double*, int)
```
```cuda
32x32 device <-> managed memory
 GPU activities:   48.06%  183.560s        70  2.62229s  104.51us  10.2285s  matrixMultiplication(double const *, double const *, double*, int)
                   47.90%  182.954s        70  2.61363s  254.42us  10.2056s  strideMatrixMultiplication(double const *, double const *, double*, int)
```
## Time comparison

![Compare CPU and CUDA](../lab4/AllLayoutsAverage.PNG)

![Compare CPU and CUDA](../lab4/3x7layout.PNG)
![Compare CPU and CUDA](../lab4/10x10layout.PNG)
![Compare CPU and CUDA](../lab4/32x32layout.PNG)

## Conclusion


