#include "cuNN.h"

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
void SBF2BFS(int db, int ds, int dbar, int df, numtype* v) {
	numtype* newv=(numtype*)malloc(db*ds*dbar*df*sizeof(numtype));
	int i=0;
	for (int b=0; b<db; b++) {
		for (int bar=0; bar<dbar; bar++) {
			for (int f=0; f<df; f++) {
				for (int s=0; s<ds; s++) {
					newv[i]=v[b*ds*dbar*df+s*dbar*df+bar*df+f];
					i++;
				}
			}
		}
	}
	memcpy(v, newv, db*ds*dbar*df*sizeof(numtype));
	free(newv);
}

sNN::sNN(int sampleLen_, int predictionLen_, int featuresCnt_, int batchCnt_, int batchSamplesCnt_, char LevelRatioS_[60], int ActivationFunction_, bool useContext_, bool useBias_) {
	pid=GetCurrentProcessId();
	tid=GetCurrentThreadId();

	batchCnt=batchCnt_;
	batchSamplesCnt=batchSamplesCnt_;
	sampleLen=sampleLen_;
	predictionLen=predictionLen_;
	featuresCnt=featuresCnt_;
	useContext=useContext_;
	useBias=useBias_;

	InputCount=sampleLen_*featuresCnt_*batchSamplesCnt;
	OutputCount=predictionLen_*featuresCnt_*batchSamplesCnt;

	//-- 0. init CUDA/BLAS
	cublasH=new void*;
	cuRandH=new void*;
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;

	//-- x. set Activation function (also sets scaleMin / scaleMax)
	setActivationFunction(ActivationFunction_);

	//-- 1. set Layout
	setLayout(LevelRatioS_);

	//-- 2. malloc neurons and weights on GPU
	if (myMalloc(&a, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&F, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&dF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&edF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&W, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&TMP, weightsCnt[0])!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_e;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_u;

	//-- device-based scalar value, to be used by reduction functions (sum, ssum, ...)
	if (myMalloc(&ss, 1)!=0) throw FAIL_MALLOC_SCALAR;

}
sNN::~sNN() {
	if (myFree(a)!=0) throw FAIL_FREE_N;
	if (myFree(F)!=0) throw FAIL_FREE_N;
	if (myFree(dF)!=0) throw FAIL_FREE_N;
	if (myFree(edF)!=0) throw FAIL_FREE_N;
	if (myFree(W)!=0) throw FAIL_FREE_W;
	if (myFree(dW)!=0) throw FAIL_FREE_W;
	if (myFree(dJdW)!=0) throw FAIL_FREE_W;
	if (myFree(TMP)!=0) throw FAIL_FREE_W;
	if (myFree(e)!=0) throw FAIL_FREE_N;
	if (myFree(u)!=0) throw FAIL_FREE_N;
	if(myFree(ss)!=0) throw FAIL_FREE_S;

	//free(weightsCnt);
	//free(a);  free(F); free(dF); free(edF);
	//free(W);
	//.....
	// free cublasH, cuRandH, curanddestroygenerator...
	free(mseT); free(mseV);

}

void sNN::setLayout(char LevelRatioS[60]) {
	int i, l, nl;
	int Levcnt;	// Levels count
	char** DescList=(char**)malloc(MAX_LEVELS*sizeof(char*)); for (i=0; i<MAX_LEVELS; i++) DescList[i]=(char*)malloc(256);

	//-- 0.1. levels and nodes count
	Levcnt = cslToArray(LevelRatioS, ',', DescList);

	for (i = 0; i<=Levcnt; i++) levelRatio[i] = (numtype)atof(DescList[i]);
	levelsCnt = (Levcnt+2);
	// set nodesCnt (single sample)
	nodesCnt[0] = InputCount;
	nodesCnt[levelsCnt-1] = OutputCount;
	for (nl = 0; nl<(levelsCnt-2); nl++) nodesCnt[nl+1] = (int)floor(nodesCnt[nl]*levelRatio[nl]);
	//-- add context neurons
	if (useContext) {
		for (nl = levelsCnt-1; nl>0; nl--) {
			nodesCnt[nl-1] += nodesCnt[nl];
		}
	}
	//-- add one bias neurons for each layer, except output layer
	if (useBias) {
		for (nl = 0; nl<(levelsCnt-1); nl++) nodesCnt[nl] += 1;
	}

	//-- 0.2. calc nodesCntTotal
	nodesCntTotal=0;
	for (l=0; l<levelsCnt; l++) nodesCntTotal+=nodesCnt[l];

	//-- 0.3. weights count
	weightsCntTotal=0;
	for (l=0; l<(levelsCnt-1); l++) {
		weightsCnt[l]=nodesCnt[l]/batchSamplesCnt*nodesCnt[l+1]/batchSamplesCnt;
		weightsCntTotal+=weightsCnt[l];
	}

	//-- 0.4. set first node and first weight for each layer
	for (l=0; l<levelsCnt; l++) {
		levelFirstNode[l]=0;
		levelFirstWeight[l]=0;
		for (int ll=0; ll<l; ll++) {
			levelFirstNode[l]+=nodesCnt[ll];
			levelFirstWeight[l]+=weightsCnt[ll];
		}
	}

	//-- ctxStart[] can only be defined after levelFirstNode has been defined.
	if (useContext) {
		for (nl=0; nl<(levelsCnt-1); nl++) {
			ctxStart[nl]=levelFirstNode[nl+1]-nodesCnt[nl+1];
		}
	}

	for (i=0; i<MAX_LEVELS; i++) free(DescList[i]);	free(DescList);

}

void sNN::setActivationFunction(int func_) {
	ActivationFunction=func_;
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		scaleMin = -1;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_EXP4:
		scaleMin = 0;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_RELU:
		scaleMin = 0;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_SOFTPLUS:
		scaleMin = 0;
		scaleMax = 1;
		break;
	default:
		scaleMin = -1;
		scaleMax = 1;
		break;
	}

}
int sNN::Activate(int level) {
	// sets dN
	int ret, retd;
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		ret=Tanh(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dTanh(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_EXP4:
		ret=Exp4(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dExp4(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_RELU:
		ret=Relu(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dRelu(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_SOFTPLUS:
		ret=SoftPlus(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dSoftPlus(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	default:
		break;
	}
	return((ret==0&&retd==0)?0:-1);
}
int sNN:: calcErr() {
	//-- sets e, bte; adds squared sum(e) to tse
	if (Vdiff(nodesCnt[levelsCnt-1], &F[levelFirstNode[levelsCnt-1]], 1, u, 1, e)!=0) return -1;	// e=F[2]-u
	if (Vssum(cublasH, nodesCnt[levelsCnt-1], e, &se, ss)!=0) return -1;							// se=ssum(e) 
	tse+=se;
	return 0;
}
int sNN::train(numtype* sample, numtype* target) {
	int l;
	char fname[MAX_PATH];
	DWORD batch_starttime, epoch_starttime;
	DWORD LDstart, LDtimeTot=0, LDcnt=0; float LDtimeAvg;
	DWORD FFstart, FFtimeTot=0, FFcnt=0; float FFtimeAvg;
	DWORD BPstart, BPtimeTot=0, BPcnt=0; float BPtimeAvg;

	DWORD training_starttime=timeGetTime();
	int epoch;
	int sc=batchSamplesCnt;

	int Ay, Ax, Astart, By, Bx, Bstart, Cy, Cx, Cstart;
	numtype* A; numtype* B; numtype* C;

	//-- Change the leading dimension in sample and target, from [Sample][Bar][Feature] to [Bar][Feature][Sample]
	int sampleCnt=batchCnt*batchSamplesCnt;	// 94

	SBF2BFS(batchCnt, batchSamplesCnt, sampleLen,  featuresCnt, sample);
	SBF2BFS(batchCnt, batchSamplesCnt, predictionLen, featuresCnt, target);

	//-- 0. Init
	
	//-- malloc mse[maxepochs], always host-side
	mseT=(numtype*)malloc(MaxEpochs*sizeof(numtype));
	mseV=(numtype*)malloc(MaxEpochs*sizeof(numtype));

	//---- 0.1. Init Neurons (must set context neurons=0, at least for layer 0)
	if( Vinit(nodesCntTotal, F, 0, 0) !=0) return -1;
	//---- the following are needed by cublas version of MbyM
	if (Vinit(nodesCntTotal, a, 0, 0)!=0) return -1;
	if (Vinit(nodesCntTotal, dF, 0, 0)!=0) return -1;
	if (Vinit(nodesCntTotal, edF, 0, 0)!=0) return -1;
	if (Vinit(weightsCntTotal, dJdW, 0, 0)!=0) return -1;

	//---- 0.2. Init W
	for (l=0; l<(levelsCnt-1); l++) VinitRnd(weightsCnt[l], &W[levelFirstWeight[l]], -1/sqrtf((numtype)nodesCnt[l]), 1/sqrtf((numtype)nodesCnt[l]), cuRandH);
	//dumpArray(weightsCntTotal, &W[0], "C:/temp/initW.txt");
	//loadArray(weightsCntTotal, &W[0], "C:/temp/initW.txt");


	//---- 0.3. Init dW
	if (Vinit(weightsCntTotal, dW, 0, 0)!=0) return -1;


	//-- 1. for every epoch, calc and display MSE
	for(epoch=0; epoch<MaxEpochs; epoch++) {

		//-- timing
		epoch_starttime=timeGetTime();

		//-- 1.0. reset epoch tse
		tse=0;

		//-- 1.1. train one batch at a time
		for (int b=0; b<batchCnt; b++) {

			//-- 1.1.1.  load samples + targets onto GPU
			LDstart=timeGetTime(); LDcnt++;
			if (loadBatchData(&F[0], &sample[b*InputCount], InputCount*sizeof(numtype) )!=0) return -1;
			if (loadBatchData(&u[0], &target[b*OutputCount], OutputCount*sizeof(numtype) )!=0) return -1;
			LDtimeTot+=((DWORD)(timeGetTime()-LDstart));

			//-- 1.1.2. Feed Forward ( W10[nc1 X nc0] X F0[nc0 X batchSize] => a1 [nc1 X batchSize] )
			FFstart=timeGetTime(); FFcnt++;
			for (l=0; l<(levelsCnt-1); l++) {
				int Ay=nodesCnt[l+1]/sc;
				int Ax=nodesCnt[l]/sc;
				numtype* A=&W[levelFirstWeight[l]];
				int By=nodesCnt[l]/sc;
				int Bx=sc;
				numtype* B=&F[levelFirstNode[l]];
				numtype* C=&a[levelFirstNode[l+1]];
				if (MbyM(cublasH, Ay, Ax, 1, false, A, By, Bx, 1, false, B, C, TMP)!=0) return -1;

				//-- activation sets F[l+1] and dF[l+1]
				if(Activate(l+1)!=0) return -1;

				//-- feed back to context neurons
				if (useContext) {
					Vcopy(nodesCnt[l+1], &F[levelFirstNode[l+1]], &F[ctxStart[l]]);
				}
			}
			FFtimeTot+=((DWORD)(timeGetTime()-FFstart));

			//-- 1.1.3. Calc Error (sets e[], te, updates tse) for the whole batch
			if (calcErr()!=0) return -1;

			//-- 1.1.4. BackPropagate, calc dJdW for the whole batch
			BPstart=timeGetTime(); BPcnt++;
			for (l = levelsCnt-1; l>0; l--) {
				if (l==(levelsCnt-1)) {
					//-- top level only
					if( VbyV2V(nodesCnt[l], e, &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = e * dF(l)
				} else {
					//-- lower levels
					Ay=nodesCnt[l+1]/sc;
					Ax=nodesCnt[l]/sc;
					Astart=levelFirstWeight[l];
					A=&W[Astart];
					By=nodesCnt[l+1]/sc;
					Bx=sc;
					Bstart=levelFirstNode[l+1];
					B=&edF[Bstart];
					Cy=Ax;	// because A gets transposed
					Cx=Bx;
					Cstart=levelFirstNode[l];
					C=&edF[Cstart];

					if (MbyM(cublasH, Ay, Ax, 1, true, A, By, Bx, 1, false, B, C, TMP)!=0) return -1;	// edF(l) = edF(l+1) * WT(l)
					if( VbyV2V(nodesCnt[l], &edF[levelFirstNode[l]], &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = edF(l) * dF(l)
				}
				
				//-- common	
				Ay=nodesCnt[l]/sc;
				Ax=sc;
				Astart=levelFirstNode[l];
				A=&edF[Astart];
				By=nodesCnt[l-1]/sc;
				Bx=sc;
				Bstart=levelFirstNode[l-1];
				B=&F[Bstart];
				Cy=Ay;
				Cx=By;// because B gets transposed
				Cstart=levelFirstWeight[l-1];
				C=&dJdW[Cstart];

				// dJdW(l-1) = edF(l) * F(l-1)
				if( MbyM(cublasH, Ay, Ax, 1, false, A, By, Bx, 1, true, B, C, TMP) !=0) return -1;

			}
			BPtimeTot+=((DWORD)(timeGetTime()-BPstart));

			//-- 1.1.5. update weights for the whole batch
			//-- W = W - LR * dJdW
			//if (Vadd(weightsCntTotal, W, 1, dJdW, -LearningRate, W)!=0) return -1;

			//-- dW = LM*dW - LR*dJdW
			if (Vdiff(weightsCntTotal, dW, LearningMomentum, dJdW, LearningRate, dW) !=0) return -1;
			//dumpArray(weightsCntTotal, dW, "C:/temp/dW.log");

			//-- W = W + dW
			if (Vadd(weightsCntTotal, W, 1, dW, 1, W)!=0) return -1;
			//dumpArray(weightsCntTotal, W, "C:/temp/W.log");

		}


		//-- 1.1. calc and display MSE
		mseT[epoch]=tse/batchCnt/nodesCnt[levelsCnt-1];
		mseV[epoch]=0;	// TO DO !
		printf("\rpid=%d, tid=%d, epoch %d, Training MSE=%f, Validation MSE=%f, duration=%d ms", pid, tid, epoch, mseT[epoch], mseV[epoch], (timeGetTime()-epoch_starttime));
		if (mseT[epoch]<TargetMSE) break;
		if((StopOnReverse && epoch>0 && mseT[epoch]>mseT[epoch-1]) ) break;
		if ((epoch%NetSaveFreq)==0) {
			//-- TO DO ! (callback?)
		}
		
	}
	ActualEpochs=epoch-((epoch>MaxEpochs)?1:0);
	float elapsed_tot=(float)timeGetTime()-(float)training_starttime;
	float elapsed_avg=elapsed_tot/ActualEpochs;
	printf("\nTraining complete. Elapsed time: %0.1f seconds. Epoch average=%0.0f ms.\n", (elapsed_tot/(float)1000), elapsed_avg);
	LDtimeAvg=(float)LDtimeTot/LDcnt; printf("LD count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", LDcnt, (LDtimeTot/(float)1000), LDtimeAvg);
	FFtimeAvg=(float)FFtimeTot/FFcnt; printf("FF count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FFcnt, (FFtimeTot/(float)1000), FFtimeAvg);
	BPtimeAvg=(float)BPtimeTot/LDcnt; printf("BP count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", BPcnt, (BPtimeTot/(float)1000), BPtimeAvg);

	//-- !!! TODO: Proper LogSaveMSE() !!!
	//dumpArray(epoch-1, mse, "C:/temp/mse.log");

	return 0;
}
int sNN::run(numtype* runW, int runSampleCnt, numtype* sample, numtype* target, numtype* Oforecast) {
	return 0;
}