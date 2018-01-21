#include "..\CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../MyTimeSeries/MyTimeSeries.h"
#include "..\cuNN\cuNN.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

void client3() {
	matrix* A=new matrix(5, 7, true, 0.1f, 0.1f);
	A->print("A");

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
	matrix* B=new matrix(5, 3, true, -0.1f, -0.1f);
	B->print("B");
	matrix* C=new matrix(7, 3);
//	MbyM(nullptr, tA->my, tA->mx, 1, false, tA->m, B->my, B->mx, 1, false, B->m, C->m);
//	C->print("C=tAxB");

	MbyM_std(7, 5, 1, false, A->m, B->my, B->mx, 1, false, B->m, C->m);
	C->print("C=AxB, MbyM_std()");

	A->X(B, C, false, false);
	C->print("C=AxB, X()");



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

	matrix* a=new matrix(2*unitsize, 3*unitsize, true, 0, 1); //a->print("a");
	matrix* b=new matrix(4*unitsize, 3*unitsize, true, 0, 1); b->print("b");
	matrix* c=new matrix(2*unitsize, 4*unitsize);
	b->transpose(); b->print(" b after transpose()");
	//MbyM_std(a->my, a->mx, 1, false, a->m, b->my, b->mx, 1, false, b->m, c->m); c->print("C-false");
	b->transpose(); b->print(" b reset");
	MbyM_std(a->my, a->mx, 1, false, a->m, b->my, b->mx, 1, true, b->m, c->m); c->print("C-true");

	//-- 0. init CUDA/BLAS
	void* cublasH=new void*;
	void* cuRandH=new void*;
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;
#ifdef USE_GPU
	//-- load a,b,c onto gpu
	numtype* da; if (cudaMalloc(&da, 3*unitsize*2*unitsize*sizeof(numtype))!=0) return;
	numtype* db; if (cudaMalloc(&db, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	numtype* dc; if (cudaMalloc(&dc, 2*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	if (cudaMemcpy(da, a->m, 2*unitsize*3*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;
	if (cudaMemcpy(db, b->m, 4*unitsize*3*unitsize*sizeof(numtype), cudaMemcpyHostToDevice)!=0) return;

	numtype* dtmp; if (cudaMalloc(&dtmp, 3*unitsize*4*unitsize*sizeof(numtype))!=0) return;
	if (MbyM(cublasH, 2*unitsize, 3*unitsize, 1, false, da, 4*unitsize, 3*unitsize, 1, true, db, dc, dtmp)!=0) return;
	if (cudaMemcpy(c->m, dc, 2*unitsize*4*unitsize*sizeof(numtype), cudaMemcpyDeviceToHost)!=0) return;
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
	int vsize=1024*1024;
	start=timeGetTime();
	if (myMalloc(&v, vsize) !=0) return -1;
	printf("myMalloc(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));
	start=timeGetTime();
	Vinit(vsize, v, 0.0f, 1.0f);
	printf("Vinit(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));
	start=timeGetTime();
	if (Vssum(vsize, v, &s)!=0) return -1;
	printf("Vssum(); elapsed time=%ld\n", (DWORD)(timeGetTime()-start));

	return(myFree(v));
}

int main() {

	//client3();	
	//client4();
	//client5();
	//client6();
	
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
	int sampleLen=20;// 20;
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

	myNN->MaxEpochs=100;
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
