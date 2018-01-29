#pragma once

#include "../CommonEnv.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

#define CUDA_BLOCK_SIZE 1024
#define MAX_STREAMS 4

EXPORT int initCUDA();
EXPORT int initCUBLAS(void* cublasH);
EXPORT int initCURand(void* cuRandH);
EXPORT int initCUstreams(void* cuStream[]);

EXPORT int Malloc_cu(numtype** var, int size);
EXPORT int Free_cu(numtype* var);

EXPORT int loadBatchData_cu(numtype* destAddr, numtype* srcAddr, int size, void* cuStream[]);
EXPORT int MbyM_cu(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T);

//-- scalar functions
EXPORT int Sadd_cu(numtype* s1, numtype* s2, numtype* ssum);

//-- vector functions;
EXPORT int Vscale_cu(int vlen, numtype* v, numtype s);
EXPORT int Vcopy_cu(int vlen, numtype* v1, numtype* v2);
EXPORT int Vadd_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vdiff_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vsum_cu(int Vlen, numtype* V, numtype* oSum, numtype* ss_d);
EXPORT int Vssum_cu(void* cublasH, int Vlen, numtype* V, numtype* oVssum, numtype* ss_d);
EXPORT int Vssum_cu_orig(int vlen, numtype* v, numtype* ovssum, numtype* ss_d);
EXPORT int Vnorm_cu(void* cublasH, int Vlen, numtype* V, numtype* oVnorm, numtype* ss_d);
EXPORT int Vinit_cu(int vlen, numtype* v, numtype start, numtype inc);
EXPORT int VbyV2V_cu(int vlen, numtype* v1, numtype* v2, numtype* ov);
EXPORT int VinitRnd_cu(int vlen, numtype* v, numtype rndmin, numtype rndmax, void* cuRandH);

//-- kernel functions wrappers
EXPORT void initGPUData(float *data, int numElements, float value);
EXPORT int dumpArray_cu(int vlen, numtype* v, const char* fname);
EXPORT int loadArray_cu(int vlen, numtype* v, const char* fname);

//-- matrix functions
EXPORT int cuMtr_cublas(void* cublasH, int my, int mx, numtype* m, numtype* otm);

EXPORT int Tanh_cu(int vlen, numtype* in, numtype* out);
EXPORT int dTanh_cu(int vlen, numtype* in, numtype* out);
EXPORT int Exp4_cu(int vlen, numtype* in, numtype* out);
EXPORT int dExp4_cu(int vlen, numtype* in, numtype* out);
EXPORT int Relu_cu(int vlen, numtype* in, numtype* out);
EXPORT int dRelu_cu(int vlen, numtype* in, numtype* out);
EXPORT int SoftPlus_cu(int vlen, numtype* in, numtype* out);
EXPORT int dSoftPlus_cu(int vlen, numtype* in, numtype* out);

EXPORT int cuMtr_naive(int my, int mx, numtype* m, numtype* omt);

