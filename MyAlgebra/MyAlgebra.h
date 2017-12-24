#pragma once

#include "../CommonEnv.h"
#include <stdio.h>
#include <math.h>

EXPORT void Mprint(int my, int mx, numtype* sm, int smy0=-1, int smx0=-1, int smy=-1, int smx=-1);
EXPORT void Msub(int my, int mx, numtype* INm, numtype* OUTsm, int smy0, int smx0, int smy, int smx);

//-- TODO: CUDA VERSIONS !!!
EXPORT void Vinit(int Vlen, int* V, int val);
EXPORT void Vinit(int Vlen, numtype* V, numtype val);
EXPORT int Vsum(int Vlen, int* V);
EXPORT numtype Vsum(numtype Vlen, numtype* V);
EXPORT void Vscale(int Vlen, int* V, float s);
EXPORT void Vscale(int Vlen, numtype* V, float s);
EXPORT void Mfill(int size, numtype* m, numtype start, numtype inc);
//--

EXPORT int Vdiff(int vlen, numtype* v1, numtype* v2, numtype* ov);
EXPORT int Vnorm(void* cublasH, int Vlen, numtype* V, numtype* oVnorm);

EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, numtype* A, int By, int Bx, numtype Bscale, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1);
EXPORT int MbyM(int Ay, int Ax, numtype Ascale, numtype* A, int By, int Bx, numtype Bscale, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1, void* cublasH=NULL);

EXPORT int myMemInit(void* cublasH);
EXPORT int myMalloc(numtype* var, int size);

EXPORT int loadBatchData(numtype* destAddr, numtype* srcAddr, int size);

EXPORT void Tanh(int Vlen, numtype* in, numtype* out);
EXPORT void dTanh(int Vlen, numtype* in, numtype* out);
EXPORT void Exp4(int Vlen, numtype* in, numtype* out);
EXPORT void dExp4(int Vlen, numtype* in, numtype* out);
EXPORT void Relu(int Vlen, numtype* in, numtype* out);
EXPORT void dRelu(int Vlen, numtype* in, numtype* out);
EXPORT void SoftPlus(int Vlen, numtype* in, numtype* out);
EXPORT void dSoftPlus(int Vlen, numtype* in, numtype* out);
