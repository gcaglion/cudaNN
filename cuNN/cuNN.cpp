#include "cuNN.h"


void SBF2BFS(int db, int ds, int dbar, int df, numtype* iv, numtype* ov) {
	//numtype* newv=(numtype*)malloc(db*ds*dbar*df*sizeof(numtype));
	int i=0;
	for (int b=0; b<db; b++) {
		for (int bar=0; bar<dbar; bar++) {
			for (int f=0; f<df; f++) {
				for (int s=0; s<ds; s++) {
					ov[i]=iv[b*ds*dbar*df+s*dbar*df+bar*df+f];
					i++;
				}
			}
		}
	}
	//memcpy(v, newv, db*ds*dbar*df*sizeof(numtype));
	//free(newv);
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

	//-- init Algebra / CUDA/CUBLAS/CURAND stuff
	Alg=new Algebra();

	//-- x. set Activation function (also sets scaleMin / scaleMax)
	setActivationFunction(ActivationFunction_);

	//-- 1. set Layout
	setLayout(LevelRatioS_, batchSamplesCnt);

	//-- 2. malloc neurons and weights on GPU
	if (myMalloc(&a, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&F, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&dF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&edF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&W, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_e;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_u;

	//-- device-based scalar value, to be used by reduction functions (sum, ssum, ...)
	if (myMalloc(&se,  1)!=0) throw FAIL_MALLOC_SCALAR;
	if (myMalloc(&tse, 1)!=0) throw FAIL_MALLOC_SCALAR;

}
sNN::~sNN() {
	myFree(a);
	myFree(F);
	myFree(dF);
	myFree(edF);
	myFree(W);
	myFree(dW);
	myFree(dJdW);
	myFree(e);
	myFree(u);
	myFree(se);
	myFree(tse);

	//free(weightsCnt);
	//free(a);  free(F); free(dF); free(edF);
	//free(W);
	free(mseT); free(mseV);

}

void sNN::setLayout(char LevelRatioS[60], int batchSamplesCnt_) {
	int i, l, nl;
	char** DescList=(char**)malloc(MAX_LEVELS*sizeof(char*)); for (i=0; i<MAX_LEVELS; i++) DescList[i]=(char*)malloc(256);

	//-- 0.1. levels and nodes count
	if (strlen(LevelRatioS)>0) {
		int Levcnt = cslToArray(LevelRatioS, ',', DescList);
		for (i = 0; i<=Levcnt; i++) levelRatio[i] = (numtype)atof(DescList[i]);
		levelsCnt = (Levcnt+2);
	}

	//-- 0.2. Input-OutputCount moved here, so can be reset when called by run()
	batchSamplesCnt=batchSamplesCnt_;
	InputCount=sampleLen*featuresCnt*batchSamplesCnt;
	OutputCount=predictionLen*featuresCnt*batchSamplesCnt;

	//-- 0.3. set nodesCnt (single sample)
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
	// sets F, dF
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
int sNN::calcErr() {
	//-- sets e, bte; adds squared sum(e) to tse
	if (Vdiff(nodesCnt[levelsCnt-1], &F[levelFirstNode[levelsCnt-1]], 1, u, 1, e)!=0) return -1;	// e=F[2]-u
	if (Vssum(nodesCnt[levelsCnt-1], e, se)!=0) return -1;											// se=ssum(e) 
	if (Vadd(1, tse, 1, se, 1, tse)!=0) return -1;													// tse+=se;
	
	return 0;
}
int sNN::FF() {
	for (int l=0; l<(levelsCnt-1); l++) {
		int Ay=nodesCnt[l+1]/batchSamplesCnt;
		int Ax=nodesCnt[l]/batchSamplesCnt;
		numtype* A=&W[levelFirstWeight[l]];
		int By=nodesCnt[l]/batchSamplesCnt;
		int Bx=batchSamplesCnt;
		numtype* B=&F[levelFirstNode[l]];
		numtype* C=&a[levelFirstNode[l+1]];

		//-- actual feed forward ( W10[nc1 X nc0] X F0[nc0 X batchSize] => a1 [nc1 X batchSize] )
		FF0start=timeGetTime(); FF0cnt++;
		if (Alg->MbyM(Ay, Ax, 1, false, A, By, Bx, 1, false, B, C)!=0) return -1;
		FF0timeTot+=((DWORD)(timeGetTime()-FF0start));

		//-- activation sets F[l+1] and dF[l+1]
		FF1start=timeGetTime(); FF1cnt++;
		if (Activate(l+1)!=0) return -1;
		FF1timeTot+=((DWORD)(timeGetTime()-FF1start));

		//-- feed back to context neurons
		//FF2start=timeGetTime(); FF2cnt++;
		if (useContext) {
			Vcopy(nodesCnt[l+1], &F[levelFirstNode[l+1]], &F[ctxStart[l]]);
		}
		//FF2timeTot+=((DWORD)(timeGetTime()-FF2start));
	}
	return 0;
}
int sNN::train(DataSet* trs) {
	int l;
	DWORD epoch_starttime;
	DWORD training_starttime=timeGetTime();
	int epoch;

	int Ay, Ax, Astart, By, Bx, Bstart, Cy, Cx, Cstart;
	numtype* A; numtype* B; numtype* C;

	//-- Change the leading dimension in sample and target, from [Sample][Bar][Feature] to [Bar][Feature][Sample]
	int sampleCnt=batchCnt*batchSamplesCnt;
	//trs->SBF2BFS(batchCnt);
	SBF2BFS(batchCnt, batchSamplesCnt, sampleLen,  featuresCnt, trs->sample, trs->sampleBFS);
	SBF2BFS(batchCnt, batchSamplesCnt, predictionLen, featuresCnt, trs->target, trs->targetBFS);

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
	for (l=0; l<(levelsCnt-1); l++) VinitRnd(weightsCnt[l], &W[levelFirstWeight[l]], -1/sqrtf((numtype)nodesCnt[l]), 1/sqrtf((numtype)nodesCnt[l]), Alg->cuRandH);
	//dumpArray(weightsCntTotal, &W[0], "C:/temp/initW.txt");
	//loadArray(weightsCntTotal, &W[0], "C:/temp/initW.txt");

	//---- 0.3. Init dW
	if (Vinit(weightsCntTotal, dW, 0, 0)!=0) return -1;

	//-- 1. for every epoch, calc and display MSE
	for(epoch=0; epoch<MaxEpochs; epoch++) {

		//-- timing
		epoch_starttime=timeGetTime();

		//-- 1.0. reset epoch tse
		if(Vinit(1, tse, 0, 0)!=0) return -1;

		//-- 1.1. train one batch at a time
		for (int b=0; b<batchCnt; b++) {

			//-- 1.1.1.  load samples + targets onto GPU
			LDstart=timeGetTime(); LDcnt++;
			if (Alg->h2d(&F[0], &trs->sampleBFS[b*InputCount], InputCount*sizeof(numtype), true )!=0) return -1;
			if (Alg->h2d(&u[0], &trs->targetBFS[b*OutputCount], OutputCount*sizeof(numtype), true )!=0) return -1;
			LDtimeTot+=((DWORD)(timeGetTime()-LDstart));

			//-- 1.1.2. Feed Forward (  )
			FFstart=timeGetTime(); FFcnt++;
			if (FF()!=0) return -1;
			FFtimeTot+=((DWORD)(timeGetTime()-FFstart));

			//-- 1.1.3. Calc Error (sets e[], te, updates tse) for the whole batch
			CEstart=timeGetTime(); CEcnt++;
			if (calcErr()!=0) return -1;
			CEtimeTot+=((DWORD)(timeGetTime()-CEstart));

			//-- 1.1.4. BackPropagate, calc dJdW for the whole batch
			BPstart=timeGetTime(); BPcnt++;
			for (l = levelsCnt-1; l>0; l--) {
				if (l==(levelsCnt-1)) {
					//-- top level only
					if( VbyV2V(nodesCnt[l], e, &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = e * dF(l)
				} else {
					//-- lower levels
					Ay=nodesCnt[l+1]/batchSamplesCnt;
					Ax=nodesCnt[l]/batchSamplesCnt;
					Astart=levelFirstWeight[l];
					A=&W[Astart];
					By=nodesCnt[l+1]/batchSamplesCnt;
					Bx=batchSamplesCnt;
					Bstart=levelFirstNode[l+1];
					B=&edF[Bstart];
					Cy=Ax;	// because A gets transposed
					Cx=Bx;
					Cstart=levelFirstNode[l];
					C=&edF[Cstart];

					if (Alg->MbyM(Ay, Ax, 1, true, A, By, Bx, 1, false, B, C)!=0) return -1;	// edF(l) = edF(l+1) * WT(l)
					if( VbyV2V(nodesCnt[l], &edF[levelFirstNode[l]], &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = edF(l) * dF(l)
				}
				
				//-- common	
				Ay=nodesCnt[l]/batchSamplesCnt;
				Ax=batchSamplesCnt;
				Astart=levelFirstNode[l];
				A=&edF[Astart];
				By=nodesCnt[l-1]/batchSamplesCnt;
				Bx=batchSamplesCnt;
				Bstart=levelFirstNode[l-1];
				B=&F[Bstart];
				Cy=Ay;
				Cx=By;// because B gets transposed
				Cstart=levelFirstWeight[l-1];
				C=&dJdW[Cstart];

				// dJdW(l-1) = edF(l) * F(l-1)
				if(Alg->MbyM(Ay, Ax, 1, false, A, By, Bx, 1, true, B, C) !=0) return -1;

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
		numtype tse_h;
		Alg->d2h(&tse_h, tse, sizeof(numtype));
		mseT[epoch]=tse_h/batchCnt/nodesCnt[levelsCnt-1];
		mseV[epoch]=0;	// TO DO !
		//printf("\rpid=%d, tid=%d, epoch %d, Training MSE=%f, Validation MSE=%f, duration=%d ms", pid, tid, epoch, mseT[epoch], mseV[epoch], (timeGetTime()-epoch_starttime));
		printf("\rpid=%d, tid=%d, epoch %d, Training MSE=%f, duration=%d ms", pid, tid, epoch, mseT[epoch], (timeGetTime()-epoch_starttime));
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
	FF0timeAvg=(float)FF0timeTot/FF0cnt; printf("FF0 count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FF0cnt, (FF0timeTot/(float)1000), FF0timeAvg);
	FF1timeAvg=(float)FF1timeTot/FF1cnt; printf("FF1 count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FF1cnt, (FF1timeTot/(float)1000), FF1timeAvg);
	//FF1atimeAvg=(float)FF1atimeTot/FF1acnt; printf("FF1a count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FF1acnt, (FF1atimeTot/(float)1000), FF1atimeAvg);
	//FF1btimeAvg=(float)FF1btimeTot/FF1bcnt; printf("FF1b count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FF1bcnt, (FF1btimeTot/(float)1000), FF1btimeAvg);
	//FF2timeAvg=(float)FF2timeTot/FF2cnt; printf("FF2 count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", FF2cnt, (FF2timeTot/(float)1000), FF2timeAvg);
	CEtimeAvg=(float)CEtimeTot/CEcnt; printf("CE count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", CEcnt, (CEtimeTot/(float)1000), CEtimeAvg);
	//VDtimeAvg=(float)VDtimeTot/VDcnt; printf("VD count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", VDcnt, (VDtimeTot/(float)1000), VDtimeAvg);
	//VStimeAvg=(float)VStimeTot/VScnt; printf("VS count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", VScnt, (VStimeTot/(float)1000), VStimeAvg);
	BPtimeAvg=(float)BPtimeTot/LDcnt; printf("BP count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", BPcnt, (BPtimeTot/(float)1000), BPtimeAvg);

	//-- !!! TODO: Proper LogSaveMSE() !!!
	//dumpArray(epoch-1, mse, "C:/temp/mse.log");

	return 0;
}
int sNN::infer(numtype* sample, numtype* Oprediction) {
	//-- Oprediction gets filled with prediction for ONE sample
	//-- sample must point to the start of the first (and only) sample to put through the network
	//-- weights must be already loaded
/*
	//-- 1. load neurons in L0 with SINGLE sample (no BATCH)
	if (Alg->h2d(&F[0], sample, InputCount*sizeof(numtype), false)!=0) return -1;
	//-- 2. Feed Forward, and copy last layer neurons (on dev) to prediction (on host)
	if (FF()!=0) return -1;
	if (Alg->d2h(Oprediction, &F[levelFirstNode[levelsCnt-1]], OutputCount*sizeof(numtype))!=0) return -1;
*/

	return 0;
}
int sNN::run(numtype* runW, DataSet* runSet) {
	//-- Oprediction gets filled with predictions for ALL samples

	//-- 1. set batchSampleCount=1, and rebuild network layout
	//setLayout("", 1);
	//-- 1b. WE ALSO NEED TO CONVERT sample/target back to SBF format!

	//-- 2. load weights (if needed)
	if (runW!=nullptr) {

	}

	//-- 3. infer prediction for every sample
/*	for (int s=0; s<runSampleCnt; s++) {
		infer(&sample[s*InputCount], &Oprediction[s*OutputCount]);

		//-- !!!!! TEMPORARY : WRITE AS-IS !!!

	}
*/
	//-- batch version
	//-- 1.1. infer one batch at a time
	for (int b=0; b<batchCnt; b++) {

		//-- 1.1.1.  load samples onto GPU
		if (Alg->h2d(&F[0], &sample[b*InputCount], InputCount*sizeof(numtype), true)!=0) return -1;

		//-- 1.1.2. Feed Forward
		if (FF()!=0) return -1;

		//-- 1.1.3. copy last layer neurons (on dev) to prediction (on host)
		if (Alg->d2h(&Oprediction[b*OutputCount], &F[levelFirstNode[levelsCnt-1]], OutputCount*sizeof(numtype))!=0) return -1;

	}

	return 0;
}