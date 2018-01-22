#include "MyAlgebra.h"

#ifdef USE_GPU
	#include "../MyCU/MyCU.h"
#endif


EXPORT void Mprint(int my, int mx, numtype* sm, const char* msg, int smy0, int smx0, int smy, int smx) {

	if (smy==-1) smy=my;
	if (smx==-1) smx=mx;

	int idx;
	if (msg!=nullptr) printf("%s [%dx%d] - from [%d,%d] to [%d,%d]\n", msg, my, mx, (smy0==-1) ? 0 : smy0, (smx0==-1) ? 0 : smx0, smy0+smy, smx0+smx);
	for (int y=0; y<smy; y++) {
		for (int x=0; x<smx; x++) {
			idx= y*mx+x;
			printf("|%2.5f", sm[idx]);
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
//-- int functions
EXPORT int Vsum(int Vlen, int* V) {
	int ret=0;
	for (int i=0; i<Vlen; i++) ret+=V[i];
	return ret;
}
EXPORT void Vscale(int Vlen, int* V, float s) {
	for (int i=0; i<Vlen; i++) V[i]=(int)(V[i]*s);
}

#ifdef USE_GPU
#else
#endif
//--

//-- scalar functions
EXPORT int Sadd(numtype* s1, numtype* s2, numtype* ssum) {
#ifdef USE_GPU
	return(Sadd_cu(s1, s2, ssum));
#else
	(*ssum)=(*s1)+(*s2);
	return 0;
#endif
}

//-- vector functions
EXPORT int Vscale(int vlen, numtype* v, numtype s) {
#ifdef USE_GPU
	return(Vscale_cu(vlen, v, s));
#else
	for (int i=0; i<vlen; i++) v[i]*=s;
	return 0;
#endif
}
EXPORT int Vcopy(int vlen, numtype* v1, numtype* v2) {
#ifdef USE_GPU
	return(Vcopy_cu(vlen, v1, v2));
#else
	for (int i=0; i<vlen; i++) v2[i]=v1[i];
	return 0;
#endif
}
EXPORT int Vadd(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov) {
#ifdef USE_GPU
	return (Vadd_cu(vlen, v1, scale1, v2, scale2, ov));
#else
	for (int i=0; i<vlen; i++) ov[i]=v2[i]*scale2+v1[i]*scale1;
	return 0;
#endif
}
EXPORT int Vdiff(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov) {
#ifdef USE_GPU
	return (Vdiff_cu(vlen, v1, scale1, v2, scale2, ov));
#else
	for (int i=0; i<vlen; i++) ov[i]=v1[i]*scale1-v2[i]*scale2;
	return 0;
#endif
}
EXPORT int Vsum(int Vlen, numtype* V, numtype* oSum, numtype* ss_d) {
	(*oSum)=0;
#ifdef USE_GPU
	return (Vsum_cu(Vlen, V, oSum, ss_d));
#else
	for (int i=0; i<Vlen; i++) (*oSum)+=V[i];
	return 0;
#endif
}
EXPORT int Vssum(void* cublasH, int Vlen, numtype* V, numtype* osSum, numtype* ss_d) {
	(*osSum)=0;
#ifdef USE_GPU
	return (Vssum_cu(cublasH, Vlen, V, osSum, ss_d));
#else
	for (int i=0; i<Vlen; i++) (*osSum)+=V[i]*V[i];
	return 0;
#endif
}
EXPORT int Vnorm(void* cublasH, int Vlen, numtype* V, numtype* oVnorm, numtype* ss_d) {
#ifdef USE_GPU
	return (Vnorm_cu(cublasH, Vlen, V, oVnorm, ss_d));
#else
	numtype vsum=0;
	for (int i=0; i<Vlen; i++) vsum+=(numtype)pow(V[i], 2);
	(*oVnorm)=(numtype)sqrt(vsum);
	return 0;
#endif
}
EXPORT int Vinit(int size, numtype* v, numtype start, numtype inc) {
#ifdef USE_GPU
	return(Vinit_cu(size, v, start, inc));
#else
	for (int i=0; i<size; i++) v[i]=start+i*inc;
	return 0;
#endif
}
EXPORT int VinitRnd(int Vlen, numtype* V, numtype rndmin, numtype rndmax, void* cuRandH) {
#ifdef USE_GPU
	return(VinitRnd_cu(Vlen, V, rndmin, rndmax, cuRandH));
#else
	time_t t;
	srand((unsigned)time(&t));

	/* Print 5 random numbers from 0 to 49 */
	for (int i = 0; i < Vlen; i++) {
		V[i] = rndmin+(numtype)rand()/((numtype)RAND_MAX+1) * (rndmax-rndmin);
		//printf("rand[%d]=%f\n", i, V[i]);
	}
/*
	unsigned int number=1234;
	int err;
	for (int i=0; i<Vlen; i++) {
		err = rand_s(&number);
		V[i] = rndmin+(numtype)number/((numtype)UINT_MAX+1) * (rndmax-rndmin);
	}
*/
	return 0;
#endif
}
EXPORT int VbyV2V(int Vlen, numtype* V1, numtype* V2, numtype* oV) {
#ifdef USE_GPU
	return(VbyV2V_cu(Vlen, V1, V2, oV));
#else
	for (int i = 0; i < Vlen; i++) oV[i] = V1[i]*V2[i];
	return 0;
#endif
}

EXPORT int Mtranspose(void* cublasH, int my, int mx, numtype* m, numtype* otm) {
#ifdef USE_GPU
	return(cuMtr_cublas(cublasH, my, mx, m, otm));
#else
	for (int y = 0; y < my; y++) {
		for (int x = 0; x < mx; x++) {
			otm[x*my+y] = m[y*mx+x];
		}
	}
	return 0;
#endif
}
//-- to fix!
int Mtranspose_old(int* my_, int* mx_, numtype* m) {
	int my=(*my_), mx=(*mx_);
	numtype** tm=(numtype**)malloc(mx*sizeof(numtype*)); for (int y=0; y<mx; y++) tm[y]=(numtype*)malloc(my*sizeof(numtype));
	for (int y = 0; y < my; y++) {
		for (int x = 0; x < mx; x++) {
			tm[x][y] = m[y*mx+x];
		}
	}
	for (int y = 0; y < my; y++) {
		for (int x = 0; x < mx; x++) {
			m[x*my+y]=tm[x][y];
		}
	}

	for (int y=0; y<mx; y++) free(tm[y]);
	free(tm);

	int tmp=my;	(*my_)=mx; (*mx_)=tmp;

	return 0;
}
EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T) {
	if (Atr) Mtranspose_old(&Ay, &Ax, A);
	if (Btr) Mtranspose_old(&By, &Bx, B);

	for (int y = 0; y < Ay; y++) {
		for (int x2 = 0; x2 < Bx; x2++) {
			C[y*Bx+x2] = 0;
			for (int x = 0; x < Ax; x++) C[y*Bx+x2] += A[y*Ax+x]*Ascale * B[x*Bx+x2]*Bscale;
		}
	}

	if (Atr) Mtranspose_old(&Ay, &Ax, A);
	if (Btr) Mtranspose_old(&By, &Bx, B);

	return 0;
}

EXPORT int MbyM(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T) {
#ifdef USE_GPU
	return MbyM_cu(cublasH, Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C, T);
#else
	return MbyM_std(Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C, T);
#endif
}

//-- memory initialization
EXPORT int myMemInit(void* cublasH, void* cuRandH) {
#ifdef USE_GPU
	if (initCUDA()!=0) return -1;
	if (initCUBLAS(cublasH)!=0) return -1;
	if (initCURand(cuRandH)!=0) return -1;
	return 0;
#else
	return 0;
#endif
}
EXPORT int myMalloc(numtype** var, int size) {
#ifdef USE_GPU
	return (Malloc_cu(var, size));
#else
	(*var) = (numtype*)malloc(size*sizeof(numtype));
	return 0;
#endif
}
EXPORT int myFree(numtype* var) {
	#ifdef USE_GPU
		return (Free_cu(var));
	#else
		free(var);
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
EXPORT void dumpData(int vlen, numtype* v, const char* fname) {
	BOOL F = HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
#ifdef USE_GPU
	dumpData_cu(vlen, v, fname);
#else
	FILE* f=fopen(fname, "w");
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", v[i]);
	fclose(f);
#endif
}

EXPORT int Tanh(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(Tanh_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)tanh(in[i]);
	return 0;
#endif 
}
EXPORT int dTanh(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return (dTanh_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1-pow(tanh(in[i]),2));
	return 0;
#endif 
}
EXPORT int Exp4(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(Exp4_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1/(1+exp(-4*in[i])));
	return 0;
#endif
}
EXPORT int dExp4(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(dExp4_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(4*exp(4*in[i])/(pow(exp(4*in[i])+1, 2)));
	return 0;
#endif
}
EXPORT int Relu(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(Relu_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(((in[i] > 0) ? 1 : 0));
	return 0;
#endif 
}
EXPORT int dRelu(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(dRelu_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(((in[i] > 0) ? in[i] : 0));
	return 0;
#endif 
}
EXPORT int SoftPlus(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(SoftPlus_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(log(1+exp(in[i])));
	return 0;
#endif 
}
EXPORT int dSoftPlus(int Vlen, numtype* in, numtype* out){
#ifdef USE_GPU 
	return(dSoftPlus_cu(Vlen, in, out));
#else 
	for (int i=0; i<Vlen; i++) out[i]=(numtype)(1/(1+exp(-in[i])));
	return 0;
#endif 
}


EXPORT int VVVcomp(int Vlen, numtype* V1, numtype* V2, numtype* oV, bool usegpu) {
#ifdef USE_GPU	
	if (usegpu) {
		if (VbyV2V_cu(Vlen, V1, V2, oV)!=0) return -1;
	} else {
		for (int i = 0; i<Vlen; i++) oV[i] = V1[i]*V2[i];
	}
#endif
	return 0;
}
EXPORT int Vssumcomp(void* cublasH, int Vlen, numtype* V, numtype* osSum, numtype* ss_d, bool usegpu) {
#ifdef USE_GPU	
	if (usegpu) {
		if (Vssum_cu(cublasH, Vlen, V, osSum, ss_d)!=0) return -1;
		//if (Vssum_cu_orig(Vlen, V, osSum, ss_d)!=0) return -1;
	} else {
		(*osSum)=0;
		for (int i=0; i<Vlen; i++) (*osSum)+=V[i]*V[i];
	}
#endif
	return 0;
}
EXPORT int Vdiffcomp(int Vlen, numtype* V1, numtype scale1, numtype* V2, numtype scale2, numtype* oV, bool usegpu) {
#ifdef USE_GPU	
	if (usegpu) {
		if (Vdiff_cu(Vlen, V1, scale1, V2, scale2, oV)!=0) return -1;
	} else {
		for (int i = 0; i<Vlen; i++) oV[i] = V1[i]*scale1-V2[i]*scale2;
	}
#endif
	return 0;
}
