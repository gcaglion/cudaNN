#include "MyAlgebra.h"

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
EXPORT int Vssum(int vlen, numtype* v, numtype* ovssum) {
	//-- if using GPU, the sum scalar also resides in GPU
#ifdef USE_GPU
	return(Vssum_cu(vlen, v, ovssum));
#else
	(*ovssum)=0;
	for (int i=0; i<vlen; i++) (*ovssum)+=v[i]*v[i];
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
EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C) {

	int m1y=Ay, m1x=Ax, m1i; numtype* m1=A;
	int m2y=By, m2x=Bx, m2i; numtype* m2=B;
	if (Atr) {
		m1y=Ax; m1x=Ay;
	}
	if (Btr) {
		m2y=Bx; m2x=By;
	}
	int mmi; numtype* mm=C;

	for (int y = 0; y < m1y; y++) {
		for (int x2 = 0; x2<m2x; x2++) {
			mmi=y*m2x+x2;
			mm[mmi]=0;
			for (int x = 0; x<m1x; x++) {
				m1i=(Atr) ? (x*m1y+y) : (y*m1x+x);
				m2i=(Btr) ? (x2*m2y+x) : (x*m2x+x2);
				mm[mmi]+=m1[m1i]*m2[m2i];
				//printf("C[%d] += A[%d] * B[%d] => %f * %f = %f\n", mmi, m1i, m2i, m1[m1i], m2[m2i], mm[mmi]);
			}
		}
	}
	//printf("\n");
	return 0;
}


//-- memory initializatin
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

//-- read/write mem<->file
EXPORT int dumpArray(int vlen, numtype* v, const char* fname) {
#ifdef USE_GPU
	return(dumpArray_cu(vlen, v, fname));
#else
	FILE* f=fopen(fname, "w");
	if (f==nullptr) return -1;
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", v[i]);
	fclose(f);
	return 0;
#endif
}
EXPORT int dumpArrayH(int vlen, numtype* v, const char* fname) {
	FILE* f=fopen(fname, "w");
	if (f==nullptr) return -1;
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", v[i]);
	fclose(f);
	return 0;
}
EXPORT int loadArray(int vlen, numtype* v, const char* fname) {
#ifdef USE_GPU
	return(loadArray_cu(vlen, v, fname));
#else
	numtype fh;
	FILE* f=fopen(fname, "r");
	if (f==nullptr) return -1;
	for (int i=0; i<vlen; i++) {
		if (fscanf(f, "%f\n", &fh)==0) return -1;
		v[i]=fh;
	}
	fclose(f);
	return 0;
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
EXPORT int MbyMcomp(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T, boolean usegpu) {
#ifdef USE_GPU	
	if (usegpu) {
		return MbyM_cu(cublasH, Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C);
	} else {
		return MbyM_std(Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C);
	}
#endif
	return 0;
}

int Vcompare(int vlen, numtype* v1, numtype* v2) {
	int ret=0;
	numtype diff;
	for (int i=0; i<vlen; i++) {
		diff=(numtype)fabs(v1[i]-v2[i]);
		if (diff>1e-5) {
			printf("diff at [%d] = %f \n", i, diff);
			ret=-1;
		}
	}
	return ret;
}

int MbyMcompare(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, int Cy, int Cx, numtype* C, numtype* T) {
#ifdef USE_GPU
	DWORD start;
	int Tsize=(Ay+By)*(Ax+Bx);

	//-- malloc host
	numtype* Ah=(numtype*)malloc(Ay*Ax*sizeof(numtype));
	numtype* Bh=(numtype*)malloc(By*Bx*sizeof(numtype));
	numtype* Ch=(numtype*)malloc(Cy*Cx*sizeof(numtype));
	numtype* Th=(numtype*)malloc(Tsize*sizeof(numtype));
	numtype* Cr=(numtype*)malloc(Cy*Cx*sizeof(numtype));	//-- gets copy of the results from device 

	//-- copy dev->host
	start=timeGetTime();
	if (cudaMemcpy(Ah, A, Ay*Ax*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	if (cudaMemcpy(Bh, B, By*Bx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	printf("copy dev->host; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	//-- cpu run
	start=timeGetTime();
	if (MbyM_std(Ay, Ax, Ascale, Atr, Ah, By, Bx, Bscale, Btr, Bh, Ch)) return -1;
	printf("CPU run; elapsed time=%ld \n", (DWORD)(timeGetTime()-start));
	//mprint(Ay, Ax, Ah, "Ah"); mprint(By, Bx, Bh, "Bh"); mprint(Cy, Cx, Ch, "Ch");

	//-- gpu run
	start=timeGetTime();
	if (MbyM_cu(cublasH, Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C)!=0) return -1;
	printf("GPU run; elapsed time=%ld \n", (DWORD)(timeGetTime()-start));

	//-- copy results dev->host
	start=timeGetTime();
	if (cudaMemcpy(Cr, C, Cy*Cx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) {
		printf("CUDA error %d\n", cudaGetLastError());
		return -1;
	}

	//-- compare results
	int ret=Vcompare(Cy*Cx, Cr, Ch);

	//-- free host
	free(Ah); free(Bh); free(Ch); free(Th); free(Cr);

	return ret;
#else
	return -1;
#endif
}

//-- class constructor/destructor
s_Algebra::s_Algebra() {
	//-- init CUDA/BLAS
	cublasH=new void*;
	cuRandH=new void*;
	for (int i=0; i<MAX_STREAMS; i++) cuStream[i]=new void*;

#ifdef USE_GPU
	if (initCUDA()!=0) throw FAIL_INITCU;
	if (initCUBLAS(cublasH)!=0) throw FAIL_INITCU;
	if (initCURand(cuRandH)!=0) throw FAIL_INITCU;
	if (initCUstreams(cuStream)!=0) throw FAIL_INITCU;
#endif
	//-- init shared scalar
	if (myMalloc(&ss, 1)!=0) throw FAIL_MALLOC_SCALAR;
}
s_Algebra::~s_Algebra() {
	myFree(ss);
	//.....
	// destroy cublasH, cuRandH, streams, curanddestroygenerator...
}
//-- class methods
int getMcol_cpu(int Ay, int Ax, numtype* A, int col, numtype* oCol) {
	for (int y=0; y<Ay; y++) oCol[y]=A[y*Ax+col];
	return 0;
}
int s_Algebra::getMcol(int Ay, int Ax, numtype* A, int col, numtype* oCol, bool forceCPU) {
#ifdef USE_GPU
	if (forceCPU) {
		return(getMcol_cpu(Ay, Ax, A, col, oCol));
	} else {
		return getMcol_cu(cublasH, Ay, Ax, A, col, oCol);
	}
#else
	return(CPUgetMcol(Ay, Ax, A, col, oCol));
#endif
}
int s_Algebra::MbyM(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, bool forceCPU) {
#ifdef USE_GPU
	if(forceCPU) {
		return(MbyM_std(Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C));
	} else {
		return (MbyM_cu(cublasH, Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C));
	}
#else
	return(MbyM_std(Ay, Ax, Ascale, Atr, A, By, Bx, Bscale, Btr, B, C));
#endif
}
int s_Algebra::h2d(numtype* destAddr, numtype* srcAddr, int size, bool useStreams) {
#ifdef USE_GPU
	return(h2d_cu(destAddr, srcAddr, size, ((useStreams)?cuStream:nullptr)) );
#else
	memcpy(destAddr, srcAddr, size);
	return 0;
#endif
}
int s_Algebra::d2h(numtype* destAddr, numtype* srcAddr, int size, bool useStreams) {
#ifdef USE_GPU
	return(d2h_cu(destAddr, srcAddr, size, ((useStreams)?cuStream:nullptr)) );
#else
	memcpy(destAddr, srcAddr, size);
	return 0;
#endif
}
