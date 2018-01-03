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
	Mfill(ay*ax, a, 0, 0.1f);
	numtype* sa=(numtype*)malloc(say*sax*sizeof(numtype));
	Msub(ay, ax, &a[sax0+ax*say0], sa, say0, sax0, say, sax);

	int by=12, bx=9;
	int sby0=4, sbx0=2;
	int sby=5, sbx=4;
	numtype* b=(numtype*)malloc(by*bx*sizeof(numtype));
	Mfill(by*bx, b, 0.0f, 0.1f);
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
	MbyM_std(ay, ax, 1, a, by, bx, 1, b, c);
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
	MbyM_std(say, sax, 1, sa, sby, sbx, 1, sb, sc);
	Mprint(scy, scx, sc);
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("a[%dx%d], start at[%dx%d], size [%dx%d] (vector math)\n", ay, ax, say0, sax0, say, sax);
	Mprint(ay, ax, &a[sax0+ax*say0], say0, sax0, say, sax);
	printf("b[%dx%d], start at[%dx%d], size [%dx%d] (vector math)\n", by, bx, sby0, sbx0, sby, sbx);
	Mprint(by, bx, &b[sbx0+bx*sby0], sby0, sbx0, sby, sbx);

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

	Vscale(levelsCnt, nodesCnt, batchSize);
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
	Mfill(nodesCntTot, N, -1, -0.1f);

	numtype* w=(numtype*)malloc(wcntTot*sizeof(numtype));
	Mfill(wcntTot, w, 1, 0.1f);
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
		MbyM(pNN->cublasH, nodesCnt[sm+2], nodesCnt[sm+1], 1, &w[s1w0], nodesCnt[sm+1], nodesCnt[sm], 1, &w[s0w0], sw_mres[sm]);
		Mprint(nodesCnt[sm+2], nodesCnt[sm], sw_mres[sm]);
		printf("\n");
	}
#endif
	return 0;
}

int main() {

	BOOL f = HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);

	//--
	tDebugInfo* DebugParms=new tDebugInfo;
	DebugParms->DebugLevel = 2;
	DebugParms->DebugDest = LOG_TO_TEXT;
	strcpy(DebugParms->fPath, "C:/temp");
	strcpy(DebugParms->fName, "Client.log");
	DebugParms->PauseOnError = 1;
	//--

	float scaleM, scaleP;

	int historyLen=100;
	int sampleLen=6;// 20;
	int predictionLen=2;
	int featuresCnt=4;	//OHLC;
	int batchSamplesCount=10;

	int totSamplesCount=historyLen-sampleLen;
	int batchCount=(int)(floor(totSamplesCount/batchSamplesCount));

	char* levelRatioS="0.5";// "1, 0.5";
/*
	int l=10;
	numtype* V=(numtype*)malloc(l*sizeof(numtype));
	VinitRnd(l, V, -0.5, 0.5);
*/
	NN* myNN=nullptr;
	try {
		myNN=new NN(sampleLen, predictionLen, featuresCnt, batchCount, batchSamplesCount, levelRatioS, false, false);
	} catch (const char* e) {
		LogWrite(DebugParms, LOG_ERROR, "NN creation failed. (%s)\n", 1, e);
	}

	myNN->setActivationFunction(NN_ACTIVATION_TANH);

	myNN->MaxEpochs=1000;
	myNN->TargetMSE=(float)0.0001;
	myNN->BP_Algo=BP_STD;
	myNN->LearningRate=(numtype)0.05;
	myNN->LearningMomentum=(numtype)0.7;

	numtype* baseData=(numtype*)malloc(featuresCnt*sizeof(numtype));
	numtype* historyData=(numtype*)malloc(historyLen*featuresCnt*sizeof(numtype));
	numtype* hd_trs=(numtype*)malloc(historyLen*featuresCnt*sizeof(numtype));
	numtype** Sample=MallocArray<numtype>(totSamplesCount, sampleLen*featuresCnt);
	numtype** Target=MallocArray<numtype>(totSamplesCount, predictionLen*featuresCnt);
	//-- flat versions
	numtype* fSample=MallocArray<numtype>(totSamplesCount * sampleLen*featuresCnt);
	numtype* fTarget=MallocArray<numtype>(totSamplesCount * predictionLen*featuresCnt);

	//-- load data ; !!!! SHOULD SET A MAX BATCHSIZE HERE, TOO, AND CYCLE THROUGH BATCHES !!!
	if (LoadFXdata(DebugParms, "EURUSD", "H1", "201508010000", historyLen, historyData, baseData)<0) return -1;
	dataTrS(historyLen, featuresCnt, historyData, baseData, DT_DELTA, myNN->scaleMin, myNN->scaleMax, hd_trs, &scaleM, &scaleP);
	//SlideArrayF(historyLen*featuresCnt, hd_trs, featuresCnt, totSamplesCount, sampleLen*featuresCnt, Sample, predictionLen*featuresCnt, Target, 2);
	fSlideArrayF(historyLen*featuresCnt, hd_trs, featuresCnt, totSamplesCount, sampleLen*featuresCnt, fSample, predictionLen*featuresCnt, fTarget, 2);

	//-- Train
	myNN->train(fSample, fTarget);

	//int ret1=client1(myNN);
	//int ret2=client2(myNN);

	system("pause");
	return 0;
}
