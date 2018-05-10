#pragma once

#include "../CommonEnv.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include "MyCUparms.h"

EXPORT Bool initCUDA();
EXPORT Bool initCUBLAS(void* cublasH);
EXPORT Bool initCURand(void* cuRandH);
EXPORT Bool initCUstreams(void* cuStream[]);

EXPORT Bool Malloc_cu(numtype** var, int size);
EXPORT Bool Free_cu(numtype* var);

//-- CPU<->GPU transfer functions
EXPORT Bool h2d_cu(numtype* destAddr, numtype* srcAddr, int size, void* cuStream[]);
EXPORT Bool d2h_cu(numtype* destAddr, numtype* srcAddr, int size, void* cuStream[]);

EXPORT Bool loadBatchData_cu(numtype* destAddr, numtype* srcAddr, int size, void* cuStream[]);
EXPORT Bool MbyM_cu(void* cublasH, int Ay, int Ax, numtype Ascale, Bool Atr, numtype* A, int By, int Bx, numtype Bscale, Bool Btr, numtype* B, numtype* C);

//-- scalar functions
EXPORT Bool Sadd_cu(numtype* s1, numtype* s2, numtype* ssum);

//-- vector functions;
EXPORT Bool getMcol_cu(void* cublasH, int Ay, int Ax, numtype* A, int col, numtype* oCol);
EXPORT Bool Vscale_cu(int vlen, numtype* v, numtype s);
EXPORT Bool Vcopy_cu(int vlen, numtype* v1, numtype* v2);
EXPORT Bool Vadd_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT Bool Vdiff_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT Bool Vsum_cu(int Vlen, numtype* V, numtype* oSum, numtype* ss_d);
EXPORT Bool Vssum_cu(int Vlen, numtype* V, numtype* oVssum);
EXPORT Bool Vnorm_cu(void* cublasH, int Vlen, numtype* V, numtype* oVnorm, numtype* ss_d);
EXPORT Bool Vinit_cu(int vlen, numtype* v, numtype start, numtype inc);
EXPORT Bool VbyV2V_cu(int vlen, numtype* v1, numtype* v2, numtype* ov);
EXPORT Bool VinitRnd_cu(int vlen, numtype* v, numtype rndmin, numtype rndmax, void* cuRandH);

//-- kernel functions wrappers
EXPORT void initGPUData(float *data, int numElements, float value);
EXPORT Bool dumpArray_cu(int vlen, numtype* v, const char* fname);
EXPORT Bool loadArray_cu(int vlen, numtype* v, const char* fname);

//-- matrix functions
EXPORT Bool cuMtr_cublas(void* cublasH, int my, int mx, numtype* m, numtype* otm);

EXPORT Bool Tanh_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool dTanh_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool Exp4_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool dExp4_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool Relu_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool dRelu_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool SoftPlus_cu(int vlen, numtype* in, numtype* out);
EXPORT Bool dSoftPlus_cu(int vlen, numtype* in, numtype* out);

EXPORT Bool cuMtr_naive(int my, int mx, numtype* m, numtype* omt);
