#include "MyAlgebra.h"

#ifdef USE_GPU
	#include "../MyCU/MyCU.h"
#endif

EXPORT void Mprint(int my, int mx, numtype* sm, int smy0, int smx0, int smy, int smx) {

	if (smy==-1) smy=my;
	if (smx==-1) smx=mx;

	int idx;
	for (int y=0; y<smy; y++) {
		for (int x=0; x<smx; x++) {
			idx= y*mx+x;
			printf("|%4.1f", sm[idx]);
		}
		printf("|\n");
	}
}
EXPORT void Msub(int my, int mx, numtype* INm, numtype* OUTsm, int smy0, int smx0, int smy, int smx) {

	if (smy==0) smy=my;
	if (smx==0) smx=mx;

	int INidx=0; int OUTidx=0;
	for (int y=0; y<smy; y++) {
		for (int x=0; x<smx; x++) {
			INidx= y*mx+x;
			OUTsm[OUTidx]=INm[INidx];
			OUTidx++;
		}
	}

}

//-- TODO: CUDA VERSIONS !!!
EXPORT void Vinit(int Vlen, int* V, int val) {
	for (int i=0; i<Vlen; i++) V[i]=val;
}
EXPORT void Vinit(int Vlen, numtype* V, numtype val) {
	for (int i=0; i<Vlen; i++) V[i]=val;
}
EXPORT int Vsum(int Vlen, int* V) {
	int ret=0;
	for (int i=0; i<Vlen; i++) ret+=V[i];
	return ret;
}
EXPORT numtype Vsum(numtype Vlen, numtype* V) {
	numtype ret=0;
	for (int i=0; i<Vlen; i++) ret+=V[i];
	return ret;
}
EXPORT void Vscale(int Vlen, int* V, float s) {
	for (int i=0; i<Vlen; i++) V[i]=(int)(V[i]*s);
}
EXPORT void Vscale(int Vlen, numtype* V, float s) {
	for (int i=0; i<Vlen; i++) V[i]*=s;
}
EXPORT void Mfill(int size, numtype* m, numtype start, numtype inc) {
	for (int i=0; i<size; i++) m[i]=start+i*inc;
}
#ifdef USE_GPU
#else
#endif
//--

EXPORT int Vdiff(int vlen, numtype* v1, numtype* v2, numtype* ov) {
#ifdef USE_GPU
	return (Vdiff_cu(vlen, v1, v2, ov));
#else
	for (int i=0; i<vlen; i++) ov[i]=v2[i]-v1[i];
	return 0;
#endif
}
EXPORT int Vnorm(void* cublasH, int Vlen, numtype* V, numtype* oVnorm) {
#ifdef USE_GPU
	return (Vnorm_cu(cublasH, Vlen, V, oVnorm));
#else
	numtype vsum=0;
	for (int i=0; i<Vlen; i++) vsum+=pow(V[i], 2);
	(*oVnorm)=(numtype)sqrt(VSSum);
	return 0;
#endif
}

EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, numtype* A, int By, int Bx, numtype Bscale, numtype* B, numtype* C,
	int sAy, int sAx, int sAy0, int sAx0,
	int sBy, int sBx, int sBy0, int sBx0,
	int sCy, int sCx, int sCy0, int sCx0
) {
	//-- As, Bs are scalars to multiply A and B cells, respectively, before multiplication
	for (int y = 0; y < Ay; y++) {
		for (int x2 = 0; x2 < Bx; x2++) {
			C[y*Bx+x2] = 0;
			for (int x = 0; x < Ax; x++) C[y*Bx+x2] += A[y*Ax+x]*Ascale * B[x*Bx+x2]*Bscale;
		}
	}
	//==== !!! sub-matrix handling missing !!! ===

	return 0;
}


EXPORT int MbyM(int Ay, int Ax, numtype Ascale, numtype* A, int By, int Bx, numtype Bscale, numtype* B, numtype* C,
	int sAy, int sAx, int sAy0, int sAx0,
	int sBy, int sBx, int sBy0, int sBx0,
	int sCy, int sCx, int sCy0, int sCx0,
	void* cublasH
) {
#ifdef USE_GPU
	return MbyM_cu(cublasH, Ay, Ax, Ascale, A, By, Bx, Bscale, B, C, sAy, sAx, sAy0, sAx0, sBy, sBx, sBy0, sBx0, sCy, sCx, sCy0, sCx0);
#else
	return MbyM_std(Ay, Ax, Ascale, A, By, Bx, Bscale, B, C, sAy, sAx, sAy0, sAx0, sBy, sBx, sBy0, sBx0, sCy, sCx, sCy0, sCx0);
#endif
}

//-- memory initialization
EXPORT int myMemInit(void* cublasH) {
#ifdef USE_GPU
	if (initCUDA()!=0) return -1;
	if (initCUBLAS(cublasH)!=0) return -1;
	return 0;
#else
	return 0;
#endif
}
EXPORT int myMalloc(numtype* var, int size) {
#ifdef USE_GPU
	return (Malloc_cu(var, size));
#else
	var = (numtype*)malloc(size*sizeof(numtype));
	return 0;
#endif
}

EXPORT int loadBatchData(numtype* destAddr, numtype* srcAddr, int size) {
#ifdef USE_GPU
	return(loadBatchData_cu(destAddr, srcAddr, size));
#else
	memcpy(destAddr, srcAddr, size);
	return 0;
#endif
}

EXPORT void Tanh(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cuTanh(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)tanh(i);
#endif 
}
EXPORT void dTanh(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cudTanh(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1-pow(tanh(i),2));
#endif 
}
EXPORT void Exp4(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cuExp4(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1/(1+exp(-4*in[i])));
#endif
}
EXPORT void dExp4(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cudExp4(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(4*exp(4*in[i])/(pow(exp(4*in[i])+1, 2)));
#endif
}
EXPORT void Relu(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cuRelu(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(((in[i] > 0) ? 1 : 0));
#endif 
}
EXPORT void dRelu(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cudRelu(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(((in[i] > 0) ? in[i] : 0));
#endif 
}
EXPORT void SoftPlus(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cuSoftPlus(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(log(1+exp(in[i])));
#endif 
}
EXPORT void dSoftPlus(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	cudSoftPlus(Vlen, in, out);
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1/(1+exp(-in[i])));
#endif 
}
