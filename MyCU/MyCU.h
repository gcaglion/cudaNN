#pragma once

#include "../CommonEnv.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

#define CUDA_BLOCK_SIZE 1024

EXPORT int initCUDA();
EXPORT int initCUBLAS(void* cublasH);
EXPORT int initCURand(void* cuRandH);

EXPORT int Malloc_cu(numtype** var, int size);

EXPORT int loadBatchData_cu(numtype* destAddr, numtype* srcAddr, int size);
EXPORT int MbyM_cu(void* cublasH, int fAy, int fAx, numtype Ascale, numtype* fA, int fBy, int fBx, numtype Bscale, numtype* fB, numtype* fC, int sAy, int sAx, int sAy0, int sAx0, int sBy, int sBx, int sBy0, int sBx0, int sCy, int sCx, int sCy0, int sCx0 );

EXPORT int Vcopy_cu(int vlen, numtype* v1, numtype* v2);
EXPORT int Vdiff_cu(int vlen, numtype* v1, numtype* v2, numtype* ov);
EXPORT int Vnorm_cu(void* cublasH, int Vlen, numtype* V, numtype* oVnorm);
EXPORT int Vinit_cu(int vlen, numtype* v, numtype val);
EXPORT int VinitRnd_cu(int vlen, numtype* v, numtype rndmin, numtype rndmax, void* cuRandH);

//-- kernel functions wrappers
EXPORT void initGPUData(float *data, int numElements, float value);
EXPORT void dumpData_cu(int vlen, numtype* v, char* fname);

EXPORT int Tanh_cu(int vlen, numtype* in, numtype* out);
EXPORT int dTanh_cu(int vlen, numtype* in, numtype* out);
EXPORT int Exp4_cu(int vlen, numtype* in, numtype* out);
EXPORT int dExp4_cu(int vlen, numtype* in, numtype* out);
EXPORT int Relu_cu(int vlen, numtype* in, numtype* out);
EXPORT int dRelu_cu(int vlen, numtype* in, numtype* out);
EXPORT int SoftPlus_cu(int vlen, numtype* in, numtype* out);
EXPORT int dSoftPlus_cu(int vlen, numtype* in, numtype* out);
