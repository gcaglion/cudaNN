#pragma once

#include "../CommonEnv.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CUDA_BLOCK_SIZE 1024

EXPORT int initCUDA();
EXPORT int initCUBLAS(void* cublasH);

EXPORT int Malloc_cu(numtype* var, int size);

EXPORT int loadBatchData_cu(numtype* destAddr, numtype* srcAddr, int size);
EXPORT int MbyM_cu(void* cublasH, int fAy, int fAx, numtype Ascale, numtype* fA, int fBy, int fBx, numtype Bscale, numtype* fB, numtype* fC, int sAy, int sAx, int sAy0, int sAx0, int sBy, int sBx, int sBy0, int sBx0, int sCy, int sCx, int sCy0, int sCx0 );

EXPORT int Vdiff_cu(int vlen, numtype* v1, numtype* v2, numtype* ov);
EXPORT int Vnorm_cu(void* cublasH, int Vlen, numtype* V, numtype* oVnorm);

//-- kernel functions wrappers
EXPORT void initGPUData(float *data, int numElements, float value);

EXPORT __global__ void cuTanh(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cudTanh(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cuExp4(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cudExp4(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cuRelu(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cudRelu(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cuSoftPlus(int Vlen, numtype* in, numtype* out);
EXPORT __global__ void cudSoftPlus(int Vlen, numtype* in, numtype* out);
