################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/matrixMul.cu 

OBJS += \
./src/matrixMul.o 

CU_DEPS += \
./src/matrixMul.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -std=c++11 -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/cuda-lab04/cuda-workspace/Lab04" -G -g -O0 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_75,code=sm_75  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -std=c++11 -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/cuda-lab04/cuda-workspace/Lab04" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_75,code=sm_75  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


