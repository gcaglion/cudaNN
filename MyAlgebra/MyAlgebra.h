#pragma once

#include "../CommonEnv.h"
#include "../MyUtils/MyUtils.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

EXPORT void Mprint(int my, int mx, numtype* sm, int smy0=-1, int smx0=-1, int smy=-1, int smx=-1);
EXPORT void Msub(int my, int mx, numtype* INm, numtype* OUTsm, int smy0, int smx0, int smy, int smx);

//-- TODO: CUDA VERSIONS !!!
EXPORT int Vsum(int Vlen, int* V);
EXPORT void Vscale(int Vlen, int* V, float s);
EXPORT void Vscale(int Vlen, numtype* V, float s);
EXPORT void Mfill(int size, numtype* m, numtype start, numtype inc);
//--

EXPORT int Vcopy(int vlen, numtype* v1, numtype* v2);
EXPORT int Vadd(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vdiff(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vssum(int Vlen, numtype* V, numtype* osSum);
EXPORT int Vnorm(void* cublasH, int Vlen, numtype* V, numtype* oVnorm);
EXPORT int Vinit(int Vlen, numtype* V, numtype val);
EXPORT int VinitRnd(int Vlen, numtype* V, numtype rndmin, numtype rndmax, void* cuRandH=NULL);
EXPORT int VbyV2V(int Vlen, numtype* V1, numtype* V2, numtype* oV);

EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1);
EXPORT int MbyM(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1);

EXPORT int myMemInit(void* cublasH, void* cuRandH);
EXPORT int myMalloc(numtype** var, int size);

EXPORT int loadBatchData(numtype* destAddr, numtype* srcAddr, int size);

EXPORT void dumpData(int vlen, numtype* v, const char* fname);

EXPORT int Tanh(int Vlen, numtype* in, numtype* out);
EXPORT int dTanh(int Vlen, numtype* in, numtype* out);
EXPORT int Exp4(int Vlen, numtype* in, numtype* out);
EXPORT int dExp4(int Vlen, numtype* in, numtype* out);
EXPORT int Relu(int Vlen, numtype* in, numtype* out);
EXPORT int dRelu(int Vlen, numtype* in, numtype* out);
EXPORT int SoftPlus(int Vlen, numtype* in, numtype* out);
EXPORT int dSoftPlus(int Vlen, numtype* in, numtype* out);
