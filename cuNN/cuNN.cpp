#include "cuNN.h"

sNN::sNN(int sampleLen_, int predictionLen_, int featuresCnt_, char LevelRatioS_[60], int* ActivationFunction_, bool useContext_, bool useBias_) {
	pid=GetCurrentProcessId();
	tid=GetCurrentThreadId();

	//-- set input and output basic dimensions (batchsize not considered yet)
	sampleLen=sampleLen_;
	predictionLen=predictionLen_;
	featuresCnt=featuresCnt_;
	useContext=useContext_;
	useBias=useBias_;

	//-- set Layout. We don't have batchSampleCnt, so we set it at 1. train() and run() will set it later
	levelRatio=(float*)malloc(60*sizeof(float));
	setLayout(LevelRatioS_, 1);
	
	//-- weights can be set now, as they are not affected by batchSampleCnt
	if (createWeights()!=0) throw FAIL_MALLOC_W;

	//-- init Algebra / CUDA/CUBLAS/CURAND stuff
	Alg=new Algebra();

	//-- x. malloc and set Activation function and scale parameters (also sets scaleMin / scaleMax)
	ActivationFunction=(int*)malloc(levelsCnt*sizeof(int));
	scaleMin=(numtype*)malloc(levelsCnt*sizeof(int));
	scaleMax=(numtype*)malloc(levelsCnt*sizeof(int));
	setActivationFunction(ActivationFunction_);

	//-- 3. malloc device-based scalar value, to be used by reduction functions (sum, ssum, ...)
	if (myMalloc(&se,  1)!=0) throw FAIL_MALLOC_SCALAR;
	if (myMalloc(&tse, 1)!=0) throw FAIL_MALLOC_SCALAR;

}
sNN::~sNN() {
	myFree(se);
	myFree(tse);

	//free(weightsCnt);
	//free(a);  free(F); free(dF); free(edF);
	//free(W);
	free(mseT); free(mseV);
	free(ActivationFunction);
	free(scaleMin); free(scaleMax);

	free(levelRatio);
	free(nodesCnt);
	free(levelFirstNode);
	free(ctxStart);
//	free(weightsCnt);
//	free(levelFirstWeight);
//	free(ActivationFunction);
}

void sNN::setLayout(char LevelRatioS[60], int batchSamplesCnt_) {
	int i, l, nl;
	char** DescList=(char**)malloc(60*sizeof(char*)); for (i=0; i<60; i++) DescList[i]=(char*)malloc(256);

	//-- 0.1. levels and nodes count
	if (strlen(LevelRatioS)>0) {
		int Levcnt = cslToArray(LevelRatioS, ',', DescList);
		for (i = 0; i<=Levcnt; i++) levelRatio[i] = (numtype)atof(DescList[i]);
		levelsCnt = (Levcnt+2);
	}

	//-- allocate level-specific parameters
	nodesCnt=(int*)malloc(levelsCnt*sizeof(int));
	levelFirstNode=(int*)malloc(levelsCnt*sizeof(int));
	ctxStart=(int*)malloc(levelsCnt*sizeof(int));
	weightsCnt=(int*)malloc((levelsCnt-1)*sizeof(int));
	levelFirstWeight=(int*)malloc((levelsCnt-1)*sizeof(int));

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

	for (i=0; i<60; i++) free(DescList[i]);	free(DescList);

}

void sNN::setActivationFunction(int* func_) {
	for (int l=0; l<levelsCnt; l++) {
		ActivationFunction[l]=func_[l];
		switch (ActivationFunction[l]) {
		case NN_ACTIVATION_TANH:
			scaleMin[l] = -1;
			scaleMax[l] = 1;
			break;
		case NN_ACTIVATION_EXP4:
			scaleMin[l] = 0;
			scaleMax[l] = 1;
			break;
		case NN_ACTIVATION_RELU:
			scaleMin[l] = 0;
			scaleMax[l] = 1;
			break;
		case NN_ACTIVATION_SOFTPLUS:
			scaleMin[l] = 0;
			scaleMax[l] = 1;
			break;
		default:
			scaleMin[l] = -1;
			scaleMax[l] = 1;
			break;
		}
	}
}
int sNN::Activate(int level) {
	// sets F, dF
	int ret, retd;
	switch (ActivationFunction[level]) {
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
		ret=-1;
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

int sNN::createNeurons() {
	//-- malloc neurons (on either CPU or GPU)
	if (myMalloc(&a, nodesCntTotal)!=0) return -1;
	if (myMalloc(&F, nodesCntTotal)!=0) return -1;
	if (myMalloc(&dF, nodesCntTotal)!=0) return -1;
	if (myMalloc(&edF, nodesCntTotal)!=0) return -1;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) return -1;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) return -1;
	//--
	if (Vinit(nodesCntTotal, F, 0, 0)!=0) return -1;
	//---- the following are needed by cublas version of MbyM
	if (Vinit(nodesCntTotal, a, 0, 0)!=0) return -1;
	if (Vinit(nodesCntTotal, dF, 0, 0)!=0) return -1;
	if (Vinit(nodesCntTotal, edF, 0, 0)!=0) return -1;
	return 0;
}
void sNN::destroyNeurons() {
	myFree(a);
	myFree(F);
	myFree(dF);
	myFree(edF);
	myFree(e);
	myFree(u);
}
int sNN::createWeights() {
	//-- malloc weights (on either CPU or GPU)
	if (myMalloc(&W, weightsCntTotal)!=0) return -1;
	if (myMalloc(&dW, weightsCntTotal)!=0) return -1;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) return -1;
	return 0;
}
void sNN::destroyWeights() {
	myFree(W);
	myFree(dW);
	myFree(dJdW);
}

int sNN::train(DataSet* trs) {
	int l;
	DWORD epoch_starttime;
	DWORD training_starttime=timeGetTime();
	int epoch;
	numtype tse_h;	// total squared error copid on host at the end of each eopch

	int Ay, Ax, Astart, By, Bx, Bstart, Cy, Cx, Cstart;
	numtype* A; numtype* B; numtype* C;

	//-- set batch count and batchSampleCnt for the network from dataset
	batchSamplesCnt=trs->batchSamplesCnt;
	batchCnt=trs->batchCnt; 
	//-- set Layout. This should not change weightsCnt[] at all, just nodesCnt[]
	setLayout("", batchSamplesCnt);

	//-- 0. malloc + init neurons
	if (createNeurons()!=0) return -1;

	//-- malloc mse[maxepochs], always host-side
	mseT=(numtype*)malloc(MaxEpochs*sizeof(numtype));
	mseV=(numtype*)malloc(MaxEpochs*sizeof(numtype));

	//---- 0.1. Init Neurons (must set context neurons=0, at least for layer 0)
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
			//dumpArray(nodesCnt[0], &F[0], "c:/temp/F0.txt");
			//dumpArray(nodesCnt[levelsCnt-1], u, "C:/temp/u.txt");
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
		Alg->d2h(&tse_h, tse, sizeof(numtype));
		mseT[epoch]=tse_h/nodesCnt[levelsCnt-1];
		mseV[epoch]=0;	// TO DO !
		//printf("\rpid=%d, tid=%d, epoch %d, Training MSE=%f, Validation MSE=%f, duration=%d ms", pid, tid, epoch, mseT[epoch], mseV[epoch], (timeGetTime()-epoch_starttime));
		printf("\rpid=%d, tid=%d, epoch %d, Training MSE=%f, duration=%d ms", pid, tid, epoch, mseT[epoch], (timeGetTime()-epoch_starttime));
		if (mseT[epoch]<TargetMSE) break;
		if((StopOnDivergence && epoch>0 && mseT[epoch]>mseT[epoch-1]) ) break;
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

	//-- test run
	if (Vinit(1, tse, 0, 0)!=0) return -1;
	for (int b=0; b<batchCnt; b++) {

		//-- load samples + targets onto GPU
		if (Alg->h2d(&F[0], &trs->sampleBFS[b*InputCount], InputCount*sizeof(numtype), true)!=0) return -1;
		if (Alg->h2d(&u[0], &trs->targetBFS[b*OutputCount], OutputCount*sizeof(numtype), true)!=0) return -1;

		//-- Feed Forward (  )
		if (FF()!=0) return -1;

		//-- Calc Error (sets e[], te, updates tse) for the whole batch
		if (calcErr()!=0) return -1;

	}
	//-- calc and display MSE
	numtype mse_h;
	Alg->d2h(&tse_h, tse, sizeof(numtype));
	mse_h=tse_h/nodesCnt[levelsCnt-1];
	printf("\nTest Run MSE=%f \n", mse_h);

	//-- feee neurons()
	destroyNeurons();

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
int sNN::run(DataSet* runSet, numtype* runW) {

	//-- set Neurons Layout based on batchSampleCount of run set
	batchSamplesCnt=runSet->batchSamplesCnt;
	batchCnt=runSet->batchCnt;
	setLayout("", runSet->batchSamplesCnt);
	
	//-- malloc + init neurons
	if (createNeurons()!=0) return -1;

	//-- reset tse=0
	if (Vinit(1, tse, 0, 0)!=0) return -1;

	//-- load weights (if needed)
	if (runW!=nullptr) {
	}

	//-- batch run
	for (int b=0; b<batchCnt; b++) {

		//-- 1.1.1.  load samples/targets onto GPU
		if (Alg->h2d(&F[0], &runSet->sampleBFS[b*InputCount], InputCount*sizeof(numtype), true)!=0) return -1;

		//-- 1.1.2. Feed Forward
		if (FF()!=0) return -1;

		//-- 1.1.3. copy last layer neurons (on dev) to prediction (on host)
		if (Alg->d2h(&runSet->predictionBFS[b*OutputCount], &F[levelFirstNode[levelsCnt-1]], OutputCount*sizeof(numtype))!=0) return -1;

		//-- 1.1.4. prediction must be converted one batch at a time
		runSet->BFS2SFB(b, runSet->targetLen, runSet->targetBFS, runSet->targetSFB);
		runSet->BFS2SFB(b, runSet->targetLen, runSet->predictionBFS, runSet->predictionSFB);

		//-- 1.1.5 copy only first-step target/prediction into target0/prediction0	- slightly better way - still room to improve
		if (Alg->getMcol(runSet->selectedFeaturesCnt, runSet->targetLen, &runSet->targetSFB[b*runSet->selectedFeaturesCnt*runSet->targetLen], 0, &runSet->target0[b*runSet->selectedFeaturesCnt], true)!=0) return -1;
		if (Alg->getMcol(runSet->selectedFeaturesCnt, runSet->targetLen, &runSet->predictionSFB[b*runSet->selectedFeaturesCnt*runSet->targetLen], 0, &runSet->prediction0[b*runSet->selectedFeaturesCnt], true)!=0) return -1;
	}


	//-- feee neurons()
	destroyNeurons();
	
	return 0;
}