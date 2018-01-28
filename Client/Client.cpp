#include "..\CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../MyTimeSeries/MyTimeSeries.h"
#include "..\cuNN\cuNN.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>

//#define cuErr(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cuErr(stat) cudaErrCheck_((stat), __FILE__, __LINE__)
boolean cudaErrCheck_(cudaError_t stat, const char *file, int line) {
	if (stat!=cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
		return true;
	}
	return false;
}

#endif

void VsumPrevs(int Vlen, int* V, int* oVsumPrevs) {
	for (int l=0; l<Vlen; l++) {
		oVsumPrevs[l]=0;
		for (int ll=0; ll<l; ll++) oVsumPrevs[l]+=V[ll];
	}
}


#ifdef USE_GPU
int client1(NN* myNN) {
	float alpha=1, beta=0;

	int ay=8, ax=12;
	int say0=2, sax0=4;
	int say=3, sax=5;
	numtype* a=(numtype*)malloc(ay*ax*sizeof(numtype));
	for(int i=0; i<(ay*ax); i++) a[i]=i*0.1f;
	numtype* sa=(numtype*)malloc(say*sax*sizeof(numtype));
	Msub(ay, ax, &a[sax0+ax*say0], sa, say0, sax0, say, sax);

	int by=12, bx=9;
	int sby0=4, sbx0=2;
	int sby=5, sbx=4;
	numtype* b=(numtype*)malloc(by*bx*sizeof(numtype));
	for (int i=0; i<(by*bx); i++) b[i]=i*0.1f;
	numtype* sb=(numtype*)malloc(sby*sbx*sizeof(numtype));
	Msub(by, bx, &b[sbx0+bx*sby0], sb, sby0, sbx0, sby, sbx);

	int cy=ay, cx=bx;
	int scy=say, scx=sbx;
	numtype* c=(numtype*)malloc(cy*cx*sizeof(numtype));
	numtype* sc=(numtype*)malloc(scy*scx*sizeof(numtype));

	//--- load a,b,c onto gpu
	numtype* a_d;
	if (cudaMalloc(&a_d, ay*ax*sizeof(numtype))!=cudaSuccess) return -1;
	if (cudaMemcpy(a_d, a, ay*ax*sizeof(numtype), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
	numtype* b_d;
	if (cudaMalloc(&b_d, by*bx*sizeof(numtype))!=cudaSuccess) return -1;
	if (cudaMemcpy(b_d, b, by*bx*sizeof(numtype), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
	numtype* c_d;
	if (cudaMalloc(&c_d, cy*cx*sizeof(numtype))!=cudaSuccess) return -1;
	if (cudaMemcpy(c_d, c, cy*cx*sizeof(numtype), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;

	printf("a[%dx%d], full\n", ay, ax);
	Mprint(ay, ax, a);
	printf("b[%dx%d], full\n", by, bx);
	Mprint(by, bx, b);

	printf("(full) C=AxB using MbM():\n");
	MbyM_std(ay, ax, 1, false, a, by, bx, 1, false, b, c);
	Mprint(cy, cx, c);
	printf("(full) C=AxB using cublasSgem():\n");
	if (cublasSgemm((*((cublasHandle_t*)myNN->cublasH)), CUBLAS_OP_N, CUBLAS_OP_N, bx, ay, ax, &alpha, b_d, bx, a_d, ax, &beta, c_d, cx)!=CUBLAS_STATUS_SUCCESS) return -1;
	if (cudaMemcpy(c, c_d, cy*cx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	Mprint(cy, cx, c);

	printf("------------------------------------------------------------------------------------------------------\n");
	printf("a[%dx%d], start at[%dx%d], size [%dx%d] (finite submatrix)\n", ay, ax, say0, sax0, say, sax);
	Mprint(say, sax, sa);
	printf("b[%dx%d], start at[%dx%d], size [%dx%d] (finite submatrix)\n", by, bx, sby0, sbx0, sby, sbx);
	Mprint(sby, sbx, sb);
	printf("(sub) C=AxB using MbM() on finite submatrices:\n");
	MbyM_std(say, sax, 1, false, sa, sby, sbx, 1, false, sb, sc);
	Mprint(scy, scx, sc);
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("a[%dx%d], start at[%dx%d], size [%dx%d] (vector math)\n", ay, ax, say0, sax0, say, sax);
	Mprint(ay, ax, &a[sax0+ax*say0], nullptr, say0, sax0, say, sax);
	printf("b[%dx%d], start at[%dx%d], size [%dx%d] (vector math)\n", by, bx, sby0, sbx0, sby, sbx);
	Mprint(by, bx, &b[sbx0+bx*sby0], nullptr, sby0, sbx0, sby, sbx);

	printf("(sub) C=AxB using cublasSgem():\n");
	if (cublasSgemm((*(cublasHandle_t*)myNN->cublasH), CUBLAS_OP_N, CUBLAS_OP_N, sbx, say, sax, &alpha, &b_d[sbx0+bx*sby0], bx, &a_d[sax0+ax*say0], ax, &beta, c_d, cx)!=CUBLAS_STATUS_SUCCESS) return -1;
	if (cudaMemcpy(c, c_d, cy*cx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	Mprint(cy, cx, c, 0, 0, scy, scx);

	return 0;
}
#endif

int client2(NN* pNN) {
	int l, sm, n;
	float alpha=1, beta=0;

	int levelsCnt=4;
	int nodesCnt[4]={ 8, 6, 4, 2 };
	int batchSize=2;

	Vscale(levelsCnt, nodesCnt, (float)batchSize);
	int nodesCntTot=Vsum(levelsCnt, nodesCnt);
	int* levelFirstNode=(int*)malloc(levelsCnt*sizeof(int));
	VsumPrevs(levelsCnt, nodesCnt, levelFirstNode);

	int* wcnt=(int*)malloc((levelsCnt-1)*sizeof(int));
	for (l=0; l<(levelsCnt-1); l++) wcnt[l]=nodesCnt[l+1]*nodesCnt[l];
	int wcntTot=Vsum(levelsCnt-1, wcnt);

	int* levelFirstWeight=(int*)malloc((levelsCnt-1)*sizeof(int));
	VsumPrevs(levelsCnt-1, wcnt, levelFirstWeight);

	printf("\n------------------------------------------------------------------------------------------------------\n");
	numtype* N=(numtype*)malloc(nodesCntTot*sizeof(numtype));
	Vinit(nodesCntTot, N, -1, -0.1f);

	numtype* w=(numtype*)malloc(wcntTot*sizeof(numtype));
	Vinit(wcntTot, w, 1, 0.1f);
	//-- sub-w[]
	numtype** sw=(numtype**)malloc((levelsCnt-1)*sizeof(numtype*));
	for (sm=0; sm<(levelsCnt-1); sm++) sw[sm]=(numtype*)malloc(wcnt[sm]*sizeof(numtype));
	l=0; n=0;
	//-- fill sw[]
	for (int i=0; i<wcntTot; i++) {
		if (n==wcnt[l]) {
			l++;
			n=0;
		}
		sw[l][n]=w[i];
		printf("l=%d, n=%d, w[%d]=%f, sw[%d][%d]=%f\n", l, n, i, w[i], l, n, sw[l][n]);
		n++;
	}
	//-- sub-w multiplications results
	numtype** sw_mres=(numtype**)malloc((levelsCnt-2)*sizeof(numtype*));
	for (sm=0; sm<(levelsCnt-2); sm++) sw_mres[sm]=(numtype*)malloc(nodesCnt[sm+2]*nodesCnt[sm]*sizeof(numtype));

	printf("\n------------------------------------------------------------------------------------------------------\n");
	for (sm=0; sm<(levelsCnt-1); sm++) {
		printf("sw[%d] [%dx%d] (finite submatrix)\n", sm, nodesCnt[sm+1], nodesCnt[sm]);
		Mprint(nodesCnt[sm+1], nodesCnt[sm], sw[sm]);
		printf("\n");
	}

	//-- MbyM_std(), multiplication on finite submatrices, into finite submatrix
	printf("\n------------------------------------------------------------------------------------------------------\n");
	for (sm=0; sm<(levelsCnt-2); sm++) {
		printf("sw[%d] X sw[%d] => sw%d%d ( [%dx%d] X [%dx%d] ) => [%dx%d] MbyM_std() - (finite subs into finite sub)\n", sm+1, sm, sm+1, sm, nodesCnt[sm+2], nodesCnt[sm+1], nodesCnt[sm+1], nodesCnt[sm], nodesCnt[sm+2], nodesCnt[sm]);
		MbyM_std(nodesCnt[sm+2], nodesCnt[sm+1], 1, false, sw[sm+1], nodesCnt[sm+1], nodesCnt[sm], 1, false, sw[sm], sw_mres[sm]);
		Mprint(nodesCnt[sm+2], nodesCnt[sm], sw_mres[sm]);
		printf("\n");
	}

	//-- MbyM_std(), multiplication on pointer submatrices, into finite submatrix
	printf("\n------------------------------------------------------------------------------------------------------\n");
	for (sm=0; sm<(levelsCnt-2); sm++) {
		int s1w0=levelFirstWeight[sm+1];
		int s0w0=levelFirstWeight[sm];
		printf("sw[%d] X sw[%d] => sw%d%d ( [%dx%d] X [%dx%d] ) => [%dx%d] MbyM_std() - (pointer subs into finite sub)\n", sm+1, sm, sm+1, sm, nodesCnt[sm+2], nodesCnt[sm+1], nodesCnt[sm+1], nodesCnt[sm], nodesCnt[sm+2], nodesCnt[sm]);
		MbyM_std(nodesCnt[sm+2], nodesCnt[sm+1], 1, false, &w[s1w0], nodesCnt[sm+1], nodesCnt[sm], 1, false, &w[s0w0], sw_mres[sm]);
		Mprint(nodesCnt[sm+2], nodesCnt[sm], sw_mres[sm]);
		printf("\n");
	}

	//-- MbyM_cu(), multiplication on pointer submatrices, into finite submatrix
#ifdef USE_GPU
	printf("\n------------------------------------------------------------------------------------------------------\n");
	for (sm=0; sm<(levelsCnt-2); sm++) {
		int s1w0=levelFirstWeight[sm+1];
		int s0w0=levelFirstWeight[sm];
		printf("sw[%d] X sw[%d] => sw%d%d ( [%dx%d] X [%dx%d] ) => [%dx%d] MbyM_cu() - (pointer subs into finite sub)\n", sm+1, sm, sm+1, sm, nodesCnt[sm+2], nodesCnt[sm+1], nodesCnt[sm+1], nodesCnt[sm], nodesCnt[sm+2], nodesCnt[sm]);
		MbyM(pNN->cublasH, nodesCnt[sm+2], nodesCnt[sm+1], 1, false, &w[s1w0], nodesCnt[sm+1], nodesCnt[sm], 1, false, &w[s0w0], sw_mres[sm], nullptr);
		Mprint(nodesCnt[sm+2], nodesCnt[sm], sw_mres[sm]);
		printf("\n");
	}
#endif
	return 0;
}

int MbyM_new(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T=nullptr) {

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
				m1i=(Atr)?(x*m1y+y):(y*m1x+x);
				m2i=(Btr)?(x2*m2y+x):(x*m2x+x2);
				mm[mmi]+=m1[m1i]*m2[m2i];
				//printf("C[%d] += A[%d] * B[%d] => %f * %f = %f\n", mmi, m1i, m2i, m1[m1i], m2[m2i], mm[mmi]);
			}
		}
	}
	//printf("\n");
	return 0;
}
void client3() {
	matrix* A=new matrix(3, 5, true, 0.1f, 0.1f);
	A->print("A");
	matrix* At=new matrix(5, 3);
	A->transposeTo(At);
	//At->print("At");


/*	matrix* sA=new matrix(2, 3);
	A->copySubTo(1, 2, sA);
	sA->print("sA");

	matrix* tA=new matrix(7, 5);
	A->transposeTo(tA);
	tA->print("tA");

	matrix* stA=new matrix(2, 3);
	tA->copySubTo(1, 2, stA);
	stA->print("stA (1)");

	sA->transposeTo(stA);
	stA->print("stA (2)");
	return;

	A->transpose();
	A->print("A after transpose()");
	matrix* tA=new matrix(7, 5);
	A->copyTo(tA);
	tA->print("tA");
*/
	matrix* B=new matrix(4, 5, true, -0.1f, -0.1f);
	B->print("B");
	matrix* Bt=new matrix(5, 4);
	B->transposeTo(Bt);
	Bt->print("Bt");
	matrix* C1=new matrix(3, 4);
	matrix* C2=new matrix(3, 4);
	//	MbyM(nullptr, tA->my, tA->mx, 1, false, tA->m, B->my, B->mx, 1, false, B->m, C->m);
//	C->print("C=tAxB");

/*	MbyM_new(At->my, At->mx, 1, false, At->m, B->my, B->mx, 1, false, B->m, C1->m);
	C1->print("C1=AtxB, MbyM_std(false, false)");
	MbyM_new(A->my, A->mx, 1, true, A->m, B->my, B->mx, 1, false, B->m, C2->m);
	C2->print("C2=AxB, MbyM_std(true, false)");
*/	MbyM_new(A->my, A->mx, 1, false, A->m, Bt->my, Bt->mx, 1, false, Bt->m, C1->m);
	C1->print("C1=AxBt, MbyM_std(false, false)");
	MbyM_new(A->my, A->mx, 1, false, A->m, B->my, B->mx, 1, true, B->m, C2->m);
	C2->print("C2=AxB, MbyM_std(false, true)");

	//A->X(B, C, false, false);
	//C->print("C=AxB, X()");



}

void client4() {
	matrix* A=new matrix(8, 5, true, 0.1f, 0.1f);
	A->print("A");
	matrix* B=new matrix(5, 12, true, -0.1f, -0.1f);
	B->print("B");
	matrix* C=new matrix(8, 12);
	MbyM(nullptr, A->my, A->mx, 1, false, A->m, B->my, B->mx, 1, false, B->m, C->m, nullptr);
	C->print("C");

	matrix* Bt=new matrix(12, 5, true, -0.1, -0.1);
	Bt->print("Bt");
	MbyM(nullptr, A->my, A->mx, 1, false, A->m, Bt->my, Bt->mx, 1, true, Bt->m, C->m, nullptr);
	C->print("C");
}

void D012_120(int d0, int d1, int d2, numtype* v) {
	numtype* newv=(numtype*)malloc(d0*d1*d2*sizeof(numtype));
	int i120, i012=0;
	for (int id0=0; id0<d0; id0++) {
		for (int id1=0; id1<d1; id1++) {
			for (int id2=0; id2<d2; id2++) {
				i120=id1*d2*d0+id2*d0+id0;
				newv[i120]=v[i012];
				i012++;
			}
		}
	}
	memcpy(v, newv, d0*d1*d2*sizeof(numtype));
	free(newv);
}
void D012_102(int d0, int d1, int d2, numtype* v) {
	numtype* newv=(numtype*)malloc(d0*d1*d2*sizeof(numtype));
	int i102, i012=0;
	for (int id0=0; id0<d0; id0++) {
		for (int id1=0; id1<d1; id1++) {
			for (int id2=0; id2<d2; id2++) {
				i102=id1*d0*d2+id0*d2+id2;
				newv[i102]=v[i012];
				i012++;
			}
		}
	}
	memcpy(v, newv, d0*d1*d2*sizeof(numtype));
	free(newv);
}

void client5() {

	int unitsize=2;

	matrix* a=new matrix(3*unitsize, 2*unitsize, true, 0, 1); a->print("a");
	matrix* b=new matrix(4*unitsize, 2*unitsize, true, 0, 1); b->print("b");
	matrix* c=new matrix(3*unitsize, 4*unitsize);
	//b->transpose(); b->print(" b after transpose()");
	//MbyM_std(a->my, a->mx, 1, false, a->m, b->my, b->mx, 1, false, b->m, c->m); c->print("C-false");
	//b->transpose(); b->print(" b reset");
	MbyM_std(a->my, a->mx, 1, false, a->m, b->my, b->mx, 1, true, b->m, c->m); c->print("C-true");

	//-- 0. init CUDA/BLAS
	void* cublasH=new void*;
	void* cuRandH=new void*;
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;
#ifdef USE_GPU
	//-- load a,b,c onto gpu
	numtype* da; if (cudaMalloc(&da, 3*unitsize*2*unitsize*sizeof(numtype))!=0) return;
	numtype* db; if (cudaMalloc(&db, 4*unitsize*2*unitsize*sizeof(numtype))!=0) return;
	numtype* dc; if (cudaMalloc(&dc, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	if (cudaMemcpy(da, a->m, 3*unitsize*2*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;
	if (cudaMemcpy(db, b->m, 4*unitsize*2*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;

	numtype* dtmp; if (cudaMalloc(&dtmp, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	if (MbyM(cublasH, 3*unitsize, 2*unitsize, 1, false, da, 4*unitsize, 2*unitsize, 1, true, db, dc, dtmp)!=0) return;
	if (cudaMemcpy(c->m, dc, 3*unitsize*4*unitsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=0) return;
	c->print("C from cublas");
#endif
/*
	matrix* bT=new matrix(3*unitsize, 4*unitsize);
	numtype* dbT; if (cudaMalloc(&dbT, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	if (Mtr(cublasH, 4*unitsize, 3*unitsize, db, dbT, 1)!=0) return;
	if (cudaMemcpy(bT->m, dbT, 3*unitsize*4*unitsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=0) return;
	bT->print(" bT after Mtr()");
*/
	/*numtype* dbT; if (cudaMalloc(&dbT, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	matrix* aT=new matrix(2*unitsize, 3*unitsize);
	matrix* bT=new matrix(4*unitsize, 3*unitsize);
	if (cudaMemcpy(da, a->m, 3*unitsize*2*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;
	if (cudaMemcpy(db, b->m, 3*unitsize*4*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;
	if (Mtranspose(a->my, a->mx, da, daT)!=0) return;
	if (Mtranspose(b->my, b->mx, db, dbT)!=0) return;
	if (cudaMemcpy(aT->m, daT, 3*unitsize*2*unitsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=0) return;
	if (cudaMemcpy(bT->m, dbT, 3*unitsize*4*unitsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=0) return;
	*/

	
}

#ifdef USE_GPU
int client6() {

	DWORD start;
	DWORD end;

	void* cublasH=new void*;
	void* cuRandH=new void*;
	
	start=timeGetTime();
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;
	printf("memInit(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	
	numtype* v;
	numtype s;
	numtype* sd; if (cudaMalloc(&sd, sizeof(numtype))!=cudaSuccess) return -1;
	int vsize=1024*1024;
	start=timeGetTime();
	if (myMalloc(&v, vsize) !=0) return -1;
	printf("myMalloc(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));
	start=timeGetTime();
	Vinit(vsize, v, 0.0f, 1.0f);
	printf("Vinit(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));
	start=timeGetTime();
	if (Vssum(cublasH, vsize, v, &s, sd)!=0) return -1;
	printf("Vssum(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	return(myFree(v));
}

int client7() {
	void* cublasH=new void*;
	void* cuRandH=new void*;
	DWORD start, end;
	bool success=true;
	numtype diff1, diff2;

	start=timeGetTime();
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;
	printf("memInit(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	int vsize= (1024*10000);
	//-- malloc host
	numtype* v1h=(numtype*)malloc(vsize*sizeof(numtype));
	numtype* v2h=(numtype*)malloc(vsize*sizeof(numtype));
	numtype* v3h=(numtype*)malloc(vsize*sizeof(numtype));
	numtype* v3r=(numtype*)malloc(vsize*sizeof(numtype));	//-- gets copy of the results from device 
	//-- malloc dev
	numtype* v1d; if (cuErr(cudaMalloc(&v1d, vsize*sizeof(numtype)))) return -1;
	numtype* v2d; if (cudaMalloc(&v2d, vsize*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* v3d; if (cudaMalloc(&v3d, vsize*sizeof(numtype))!=cudaSuccess) return -1;
	//-- sum variables (dev and host)
	numtype s1, s2, s1h, s2h, s1d, s2d, diffh;
	numtype* ssd; if (cudaMalloc(&ssd, sizeof(numtype))!=cudaSuccess) return -1;

	for (int test=0; test<100; test++) {
		
		//-- init dev
		start=timeGetTime();
		if (VinitRnd(vsize, v1d, -1, 1, cuRandH)!=0) return -1;
		if (VinitRnd(vsize, v2d, -1, 1, cuRandH)!=0) return -1;
		printf("Init dev; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- copy dev->host
		start=timeGetTime();
		if (cudaMemcpy(v1h, v1d, vsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		if (cudaMemcpy(v2h, v2d, vsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		printf("copy dev->host; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- cpu run
		start=timeGetTime();
		//if (VVVcomp(vsize, v1h, v2h, v3h, false)!=0) return -1;
		//if (Vssumcomp(cublasH, vsize, v1h, &s1, ssd, false)!=0) return -1;
		//if (Vssumcomp(cublasH, vsize, v2h, &s2, ssd, false)!=0) return -1;
		if (Vdiffcomp(vsize, v1h, 1, v2h, 1, v3h, false)!=0) return -1;
		//s1h=s1; s2h=s2;
		printf("CPU run; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- gpu run
		start=timeGetTime();
		//if (VVVcomp(vsize, v1d, v2d, v3d, true)!=0) return -1;
		//if (Vssumcomp(cublasH, vsize, v1d, &s1, ssd, true)!=0) return -1;
		//if (Vssumcomp(cublasH, vsize, v2d, &s2, ssd, true)!=0) return -1;
		if (Vdiffcomp(vsize, v1d, 1, v2d, 1, v3d, true)!=0) return -1;
		//s1d=s1; s2d=s2;
		printf("GPU run; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- copy results dev->host, and compare
		start=timeGetTime();
		if (cudaMemcpy(v3r, v3d, vsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		for (int i=0; i<vsize; i++) {
			if (v3r[i]!=v3h[i]) {
				success=false;
				break;
			}
		}

		//-- compare results
		//success=(diffh==diffd);	// (s1d==s1h &&s2d==s2h);
		//diff1=fabs(s1d-s1h); diff2=fabs(s2d-s2h);
		//numtype diffdiff=fabs(diffh-diffd);
		printf("Result: %s\n", (success) ? "SUCCESS" : "FAILURE");

	}
}
void mprint(int my, int mx, numtype* m, char* msg=nullptr, int smy0=-1, int smx0=-1, int smy=-1, int smx=-1) {
	if (smy==-1) smy=my;
	if (smx==-1) smx=mx;

	int idx;
	if (msg!=nullptr) printf("%s [%dx%d] - from [%d,%d] to [%d,%d]\n", msg, my, mx, (smy0==-1) ? 0 : smy0, (smx0==-1) ? 0 : smx0, smy0+smy, smx0+smx);
	for (int y=0; y<smy; y++) {
		for (int x=0; x<smx; x++) {
			idx= y*mx+x;
			printf("|%4.1f", m[idx]);
		}
		printf("|\n");
	}

}
int client8() {
	void* cublasH=new void*;
	void* cuRandH=new void*;
	DWORD start, end;
	bool success=true;

	start=timeGetTime();
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;
	printf("memInit(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	int Ay=3, Ax=2; bool trA=false;
	int By=4, Bx=2; bool trB=true;
	int Cy=Ay, Cx=Bx;

	int sizemult=200;
	Ay*=sizemult; Ax*=sizemult;
	By*=sizemult; Bx*=sizemult;
	Cy*=sizemult; Cx*=sizemult;


	//-- malloc host
	numtype* Ah=(numtype*)malloc(Ay*Ax*sizeof(numtype));
	numtype* Bh=(numtype*)malloc(By*Bx*sizeof(numtype));
	numtype* Ch=(numtype*)malloc(Cy*Cx*sizeof(numtype));
	numtype* Th=(numtype*)malloc((Ay+By)*(Ax+Bx)*sizeof(numtype));
	numtype* Cr=(numtype*)malloc(Cy*Cx*sizeof(numtype));	//-- gets copy of the results from device 
															//-- malloc dev
	numtype* Ad; if (cudaMalloc(&Ad, Ay*Ax*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Bd; if (cudaMalloc(&Bd, By*Bx*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Cd; if (cudaMalloc(&Cd, Cy*Cx*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Td; if (cudaMalloc(&Td, (Ay+By)*(Ax+Bx)*sizeof(numtype))!=cudaSuccess) return -1;

	for (int test=0; test<10; test++) {

		//-- init dev
		start=timeGetTime();
		if (VinitRnd(Ay*Ax, Ad, -1, 1, cuRandH)!=0) return -1;
		if (VinitRnd(By*Bx, Bd, -1, 1, cuRandH)!=0) return -1;
		if (Vinit(Cy*Cx, Cd, 0, 0)!=0) return -1;
		printf("Init dev; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- copy dev->host
		start=timeGetTime();
		if (cudaMemcpy(Ah, Ad, Ay*Ax*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		if (cudaMemcpy(Bh, Bd, By*Bx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		//if (cudaMemcpy(Ch, Cd, Cy*Cx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
		printf("copy dev->host; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- cpu run
		start=timeGetTime();
		if (MbyMcomp(cublasH, Ay, Ax, 1, trA, Ah, By, Bx, 1, trB, Bh, Ch, Th, false)!=0) return -1;
		printf("CPU run; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));
		//mprint(Ay, Ax, Ah, "Ah"); mprint(By, Bx, Bh, "Bh"); mprint(Cy, Cx, Ch, "Ch");

		//-- gpu run
		start=timeGetTime();
		if (MbyMcomp(cublasH, Ay, Ax, 1, trA, Ad, By, Bx, 1, trB, Bd, Cd, Td, true)!=0) return -1;
		printf("GPU run; elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

		//-- copy results dev->host, and compare
		start=timeGetTime();
		if (cudaMemcpy(Cr, Cd, Cy*Cx*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) {
			printf("CUDA error %d\n", cudaGetLastError());
			return -1;
		}
		//mprint(Cy, Cx, Cr, "Cr");
		numtype diff;
		success=true;
		for (int i=0; i<(Cy*Cx); i++) {
			diff=fabs(Cr[i]-Ch[i]);
			if (diff>1e-5) {
				printf("test=%d: diff at [%d] = %f \n", test, i, diff);
				success=false;
				//break;
			}
		}

		//-- compare results
		//success=(diffh==diffd);	// (s1d==s1h &&s2d==s2h);
		//diff1=fabs(s1d-s1h); diff2=fabs(s2d-s2h);
		//numtype diffdiff=fabs(diffh-diffd);
		printf("Result: %s\n", (success) ? "SUCCESS" : "FAILURE");

	}
}
int client9() {
	void* cublasH=new void*;
	void* cuRandH=new void*;
	DWORD start, end;

	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;

	int Ay=2, Ax=3; bool trA=true;
	int By=2, Bx=4; bool trB=false;

	int Cy=(trA) ? Ax : Ay;
	int Cx=(trB) ? By : Bx;

	int sizemult=200;
	Ay*=sizemult; Ax*=sizemult;
	By*=sizemult; Bx*=sizemult;
	Cy*=sizemult; Cx*=sizemult;


	//-- malloc dev
	numtype* Ad; if (cudaMalloc(&Ad, Ay*Ax*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Bd; if (cudaMalloc(&Bd, By*Bx*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Cd; if (cudaMalloc(&Cd, Cy*Cx*sizeof(numtype))!=cudaSuccess) return -1;
	numtype* Td; if (cudaMalloc(&Td, (Ay+By)*(Ax+Bx)*sizeof(numtype))!=cudaSuccess) return -1;

	int ret;
	for (int test=0; test<10; test++) {

		//-- init dev
		if (VinitRnd(Ay*Ax, Ad, -1, 1, cuRandH)!=0) return -1;
		if (VinitRnd(By*Bx, Bd, -1, 1, cuRandH)!=0) return -1;

		start=timeGetTime();
		//-- run test
		ret = MbyMcompare(cublasH, Ay, Ax, 1, trA, Ad, By, Bx, 1, trB, Bd, Cy, Cx, Cd, Td);
		printf("Test %d %s\n", test, (ret==0) ? "SUCCESS" : "FAILURE");

	}

	//-- free dev
	if (cudaFree(Ad)!=cudaSuccess) return -1;
	if (cudaFree(Bd)!=cudaSuccess) return -1;
	if (cudaFree(Cd)!=cudaSuccess) return -1;
	if (cudaFree(Td)!=cudaSuccess) return -1;

	return 0;
}

#endif

int main() {

	//client3();	
	//client4();
	//client5();
	//client6();
	//client7();
	//client8();
	//client9();
	//system("pause");
	//return -1;

	//--
	tDebugInfo* DebugParms=new tDebugInfo;
	DebugParms->DebugLevel = 2;
	DebugParms->DebugDest = LOG_TO_TEXT;
	strcpy(DebugParms->fPath, "C:/temp");
	strcpy(DebugParms->fName, "Client.log");
	DebugParms->PauseOnError = 1;
	//--

	float scaleM, scaleP;

	int historyLen=1000;
	int sampleLen=20;
	int predictionLen=2;
	int featuresCnt=4;	//OHLC !!! FIXED !!! (it's hard-coded in LoadFxData);
	int batchSamplesCount=10;

	int totSamplesCount=historyLen-sampleLen;
	int batchCount=(int)(floor(totSamplesCount/batchSamplesCount));

	char* levelRatioS="0.5, 1";// "1, 0.5";

	NN* myNN=nullptr;
	try {
		myNN=new NN(sampleLen, predictionLen, featuresCnt, batchCount, batchSamplesCount, levelRatioS, true, false);
	} catch (const char* e) {
		LogWrite(DebugParms, LOG_ERROR, "NN creation failed. (%s)\n", 1, e);
	}

	myNN->setActivationFunction(NN_ACTIVATION_TANH);

	myNN->MaxEpochs=200;
	myNN->TargetMSE=(float)0.0001;
	myNN->BP_Algo=BP_STD;
	myNN->LearningRate=(numtype)0.005;
	myNN->LearningMomentum=(numtype)0.7;

	numtype* baseData=(numtype*)malloc(featuresCnt*sizeof(numtype));
	numtype* historyData=(numtype*)malloc(historyLen*featuresCnt*sizeof(numtype));
	numtype* hd_trs=(numtype*)malloc(historyLen*featuresCnt*sizeof(numtype));

	numtype* fSample=MallocArray<numtype>(totSamplesCount * sampleLen*featuresCnt);
	numtype* fTarget=MallocArray<numtype>(totSamplesCount * predictionLen*featuresCnt);

	//-- load data ; !!!! SHOULD SET A MAX BATCHSIZE HERE, TOO, AND CYCLE THROUGH BATCHES !!!
	if (LoadFXdata(DebugParms, "EURUSD", "H1", "201508010000", historyLen, historyData, baseData)<0) return -1;
	dataTrS(historyLen, featuresCnt, historyData, baseData, DT_DELTA, myNN->scaleMin, myNN->scaleMax, hd_trs, &scaleM, &scaleP);

	fSlideArrayF(historyLen*featuresCnt, hd_trs, featuresCnt, totSamplesCount, sampleLen*featuresCnt, fSample, predictionLen*featuresCnt, fTarget, 2);

	//-- Train
	myNN->train(fSample, fTarget);

	system("pause");
	return 0;
}
