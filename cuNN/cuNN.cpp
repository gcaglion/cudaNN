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
	if (!createWeights()) throw FAIL_MALLOC_W;

	//-- init Algebra / CUDA/CUBLAS/CURAND stuff
	Alg=new Algebra();

	//-- x. malloc and set Activation function and scale parameters (also sets scaleMin / scaleMax)
	ActivationFunction=(int*)malloc(levelsCnt*sizeof(int));
	scaleMin=(numtype*)malloc(levelsCnt*sizeof(int));
	scaleMax=(numtype*)malloc(levelsCnt*sizeof(int));
	setActivationFunction(ActivationFunction_);

	//-- 3. malloc device-based scalar value, to be used by reduction functions (sum, ssum, ...)
	if (myMalloc(&se, 1)!=0) throw FAIL_MALLOC_SCALAR;
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
			ctxStart[nl]=levelFirstNode[nl+1]-nodesCnt[nl+1]+((useBias) ? 1 : 0);
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

bool sNN::FF() {
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
		if (Alg->MbyM(Ay, Ax, 1, false, A, By, Bx, 1, false, B, C)!=0) return false;
		FF0timeTot+=((DWORD)(timeGetTime()-FF0start));

		//-- activation sets F[l+1] and dF[l+1]
		FF1start=timeGetTime(); FF1cnt++;
		if (!Activate(l+1)) return false;
		FF1timeTot+=((DWORD)(timeGetTime()-FF1start));

		//-- feed back to context neurons
		FF2start=timeGetTime(); FF2cnt++;
		if (useContext) {
			Vcopy(nodesCnt[l+1], &F[levelFirstNode[l+1]], &F[ctxStart[l]]);
		}
		FF2timeTot+=((DWORD)(timeGetTime()-FF2start));
	}
	return true;
}
bool sNN::Activate(int level) {
	// sets F, dF
	int ret, retd;
	int skipBias=(useBias&&level!=(levelsCnt-1))?1:0;	//-- because bias neuron does not exits in outer layer
	int nc=nodesCnt[level]-skipBias;
	numtype* va=&a[levelFirstNode[level]+skipBias];
	numtype* vF=&F[levelFirstNode[level]+skipBias];
	numtype* vdF=&dF[levelFirstNode[level]+skipBias];

	switch (ActivationFunction[level]) {
	case NN_ACTIVATION_TANH:
		ret=Tanh(nc, va, vF);
		retd=dTanh(nc, va, vdF);
		break;
	case NN_ACTIVATION_EXP4:
		ret=Exp4(nc, va, vF);
		retd=dExp4(nc, va, vdF);
		break;
	case NN_ACTIVATION_RELU:
		ret=Relu(nc, va, vF);
		retd=dRelu(nc, va, vdF);
		break;
	case NN_ACTIVATION_SOFTPLUS:
		ret=SoftPlus(nc, va, vF);
		retd=dSoftPlus(nc, va, vdF);
		break;
	default:
		ret=-1;
		break;
	}
	return(ret==0&&retd==0);
}
bool sNN::calcErr() {
	//-- sets e, bte; adds squared sum(e) to tse
	if (Vdiff(nodesCnt[levelsCnt-1], &F[levelFirstNode[levelsCnt-1]], 1, u, 1, e)!=0) return false;	// e=F[2]-u
	if (Vssum(nodesCnt[levelsCnt-1], e, se)!=0) return false;											// se=ssum(e) 
	if (Vadd(1, tse, 1, se, 1, tse)!=0) return false;													// tse+=se;

	return true;
}

bool sNN::mallocNeurons() {
	//-- malloc neurons (on either CPU or GPU)
	if (myMalloc(&a, nodesCntTotal)!=0) return false;
	if (myMalloc(&F, nodesCntTotal)!=0) return false;
	if (myMalloc(&dF, nodesCntTotal)!=0) return false;
	if (myMalloc(&edF, nodesCntTotal)!=0) return false;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) return false;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) return false;
	return true;
}
bool sNN::initNeurons(){
	//--
	if (Vinit(nodesCntTotal, F, 0, 0)!=0) return false;
	//---- the following are needed by cublas version of MbyM
	if (Vinit(nodesCntTotal, a, 0, 0)!=0) return false;
	if (Vinit(nodesCntTotal, dF, 0, 0)!=0) return false;
	if (Vinit(nodesCntTotal, edF, 0, 0)!=0) return false;

	if (useBias) {
		for (int l=0; l<(levelsCnt-1); l++) {
			//-- set every bias node's F=1
			if (Vinit(1, &F[levelFirstNode[l]], 1, 0)!=0) return false;
		}
	}

	return true;
}
void sNN::destroyNeurons() {
	myFree(a);
	myFree(F);
	myFree(dF);
	myFree(edF);
	myFree(e);
	myFree(u);
}
bool sNN::createWeights() {
	//-- malloc weights (on either CPU or GPU)
	if (myMalloc(&W, weightsCntTotal)!=0) return false;
	if (myMalloc(&prevW, weightsCntTotal)!=0) return false;
	if (myMalloc(&dW, weightsCntTotal)!=0) return false;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) return false;
	return true;
}
void sNN::destroyWeights() {
	myFree(W);
	myFree(prevW);
	myFree(dW);
	myFree(dJdW);
}

bool sNN::BP_std(){
	int Ay, Ax, Astart, By, Bx, Bstart, Cy, Cx, Cstart;
	numtype* A; numtype* B; numtype* C;

	for (int l = levelsCnt-1; l>0; l--) {
		if (l==(levelsCnt-1)) {
			//-- top level only
			if (VbyV2V(nodesCnt[l], e, &dF[levelFirstNode[l]], &edF[levelFirstNode[l]])!=0) return false;	// edF(l) = e * dF(l)
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

			if (Alg->MbyM(Ay, Ax, 1, true, A, By, Bx, 1, false, B, C)!=0) return false;	// edF(l) = edF(l+1) * WT(l)
			if (VbyV2V(nodesCnt[l], &edF[levelFirstNode[l]], &dF[levelFirstNode[l]], &edF[levelFirstNode[l]])!=0) return false;	// edF(l) = edF(l) * dF(l)
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
		if (Alg->MbyM(Ay, Ax, 1, false, A, By, Bx, 1, true, B, C)!=0) return false;

	}
	return true;
}
bool sNN::WU_std(){

	//-- 1. calc dW = LM*dW - LR*dJdW
	if (Vdiff(weightsCntTotal, dW, LearningMomentum, dJdW, LearningRate, dW)!=0) return false;

	//-- 2. update W = W + dW for current batch
	if (Vadd(weightsCntTotal, W, 1, dW, 1, W)!=0) return false;

	return true;
}
bool sNN::ForwardPass(DataSet* ds, int batchId, bool haveTargets) {

	//-- 1. load samples (and targets, if passed) from single batch in dataset onto input layer
	LDstart=timeGetTime(); LDcnt++;
	if (Alg->h2d(&F[(useBias)?1:0], &ds->sampleBFS[batchId*InputCount], InputCount*sizeof(numtype), true)!=0) return false;
	if (haveTargets) {
		if (Alg->h2d(&u[0], &ds->targetBFS[batchId*OutputCount], OutputCount*sizeof(numtype), true)!=0) return false;
	}
	LDtimeTot+=((DWORD)(timeGetTime()-LDstart));

	//-- 2. Feed Forward
	FFstart=timeGetTime(); FFcnt++;
	if (!FF()) return false;
	FFtimeTot+=((DWORD)(timeGetTime()-FFstart));

	//-- 3. If we have targets, Calc Error (sets e[], te, updates tse) for the whole batch
	CEstart=timeGetTime(); CEcnt++;
	if (haveTargets) {
		if (!calcErr()) return false;
	}
	CEtimeTot+=((DWORD)(timeGetTime()-CEstart));

	return true;
}
bool sNN::BackwardPass(DataSet* ds, int batchId, bool updateWeights) {

	//-- 1. BackPropagate, calc dJdW for for current batch
	BPstart=timeGetTime(); BPcnt++;
	if (!BP_std()) return false;
	BPtimeTot+=((DWORD)(timeGetTime()-BPstart));

	//-- 2. Weights Update for current batch
	WUstart=timeGetTime(); WUcnt++;
	if (updateWeights) {
		if (!WU_std()) return false;
	}
	WUtimeTot+=((DWORD)(timeGetTime()-WUstart));

	return true;
}
bool sNN::epochMetCriteria(int epoch, DWORD starttime, bool displayProgress) {
	numtype tse_h;	// total squared error copid on host at the end of each eopch

	Alg->d2h(&tse_h, tse, sizeof(numtype));
	mseT[epoch]=tse_h/nodesCnt[levelsCnt-1]/batchCnt;
	mseV[epoch]=0;	// TO DO !
	if(displayProgress) printf("\rpid=%d, tid=%d, epoch %d, Training TSE=%f, MSE=%1.10f, duration=%d ms", pid, tid, epoch, tse_h, mseT[epoch], (timeGetTime()-starttime));
	if (mseT[epoch]<TargetMSE) return true;
	if ((StopOnDivergence && epoch>1&&mseT[epoch]>mseT[epoch-1])) return true;
	if ((epoch%NetSaveFreq)==0) {
		//-- TO DO ! (callback?)
	}

	return false;
}
int sNN::train(DataSet* trs) {
	int l;
	DWORD epoch_starttime;
	DWORD training_starttime=timeGetTime();
	int epoch, b;

	//-- set batch count and batchSampleCnt for the network from dataset
	batchSamplesCnt=trs->batchSamplesCnt;
	batchCnt=trs->batchCnt;
	//-- set Layout. This should not change weightsCnt[] at all, just nodesCnt[]
	setLayout("", batchSamplesCnt);

	//-- 0. malloc + init neurons
	if (!mallocNeurons()) return -1;
	if (!initNeurons()) return -1;

	//-- malloc mse[maxepochs], always host-side
	mseT=(numtype*)malloc(MaxEpochs*sizeof(numtype));
	mseV=(numtype*)malloc(MaxEpochs*sizeof(numtype));

	//---- 0.2. Init W
	for (l=0; l<(levelsCnt-1); l++) VinitRnd(weightsCnt[l], &W[levelFirstWeight[l]], -1/sqrtf((numtype)nodesCnt[l]), 1/sqrtf((numtype)nodesCnt[l]), Alg->cuRandH);
	//if(!dumpArray(weightsCntTotal, &W[0], "C:/temp/initW.txt")) return -1;
	if(!loadArray(weightsCntTotal, &W[0], "C:/temp/initW.txt")) return -1;

	//---- 0.3. Init dW, dJdW
	if (Vinit(weightsCntTotal, dW, 0, 0)!=0) return -1;
	if (Vinit(weightsCntTotal, dJdW, 0, 0)!=0) return -1;

	//-- 1. for every epoch, train all batch with one Forward pass ( loadSamples(b)+FF()+calcErr() ), and one Backward pass (BP + calcdW + W update)
	for (epoch=0; epoch<MaxEpochs; epoch++) {

		//-- timing
		epoch_starttime=timeGetTime();

		//-- 1.0. reset epoch tse
		if (Vinit(1, tse, 0, 0)!=0) return -1;

		//-- 1.1. train one batch at a time
		for (b=0; b<batchCnt; b++) {

			//-- forward pass, with targets
			if (!ForwardPass(trs, b, true)) return -1;

			//-- backward pass, with weights update
			if (!BackwardPass(trs, b, true)) return -1;

		}

		//-- 1.2. calc and display epoch MSE (for ALL batches), and check criteria for terminating training (targetMSE, Divergence)
		if (epochMetCriteria(epoch, epoch_starttime)) break;

	}
	ActualEpochs=epoch-((epoch>MaxEpochs)?1:0);

	//-- 2. test run. need this to make sure all batches pass through the net with the latest weights, and training targets
	TRstart=timeGetTime(); TRcnt++;
	if (Vinit(1, tse, 0, 0)!=0) return -1;
	for (b=0; b<batchCnt; b++) if(!ForwardPass(trs, b, true)) return -1;
	TRtimeTot+=((DWORD)(timeGetTime()-TRstart));

	//-- calc and display final epoch MSE
	printf("\n"); epochMetCriteria(ActualEpochs-1, epoch_starttime);


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
	TRtimeAvg=(float)TRtimeTot/LDcnt; printf("TR count=%d ; time-tot=%0.1f s. time-avg=%0.0f ms.\n", TRcnt, (TRtimeTot/(float)1000), TRtimeAvg);


	//-- feee neurons()
	destroyNeurons();

	return 0;
}
int sNN::run(DataSet* runSet, numtype* runW) {

	//-- set Neurons Layout based on batchSampleCount of run set
	batchSamplesCnt=runSet->batchSamplesCnt;
	batchCnt=runSet->batchCnt;
	setLayout("", runSet->batchSamplesCnt);

	//-- malloc + init neurons
	if (!mallocNeurons()) return -1;
	if (!initNeurons()) return -1;

	//-- reset tse=0
	if (Vinit(1, tse, 0, 0)!=0) return -1;

	//-- load weights (if needed)
	if (runW!=nullptr) {
	}

	//-- batch run
	for (int b=0; b<batchCnt; b++) {

		//-- 1.1.1.  load samples/targets onto GPU
		if (Alg->h2d(&F[(useBias)?1:0], &runSet->sampleBFS[b*InputCount], InputCount*sizeof(numtype), true)!=0) return -1;
		if (Alg->h2d(&u[0], &runSet->targetBFS[b*OutputCount], OutputCount*sizeof(numtype), true)!=0) return -1;

		//-- 1.1.2. Feed Forward
		if (!FF()) return -1;

		//-- 1.1.3. copy last layer neurons (on dev) to prediction (on host)
		if (Alg->d2h(&runSet->predictionBFS[b*OutputCount], &F[levelFirstNode[levelsCnt-1]], OutputCount*sizeof(numtype))!=0) return -1;

		calcErr();
	}

	//-- calc and display final epoch MSE
	numtype tse_h;	// total squared error copid on host at the end of the run
	Alg->d2h(&tse_h, tse, sizeof(numtype));
	numtype mseR=tse_h/nodesCnt[levelsCnt-1]/batchCnt;
	printf("\npid=%d, tid=%d, Run final MSE=%f\n", pid, tid, mseR);

	//-- convert prediction from BFS to SFB (fol all batches at once)
	runSet->BFS2SFBfull(runSet->targetLen, runSet->predictionBFS, runSet->predictionSFB);
	//-- extract first bar only from target/prediction SFB
	if (Alg->getMcol(runSet->batchCnt*runSet->batchSamplesCnt*runSet->selectedFeaturesCnt, runSet->targetLen, runSet->targetSFB, 0, runSet->target0, true)!=0) return -1;
	if (Alg->getMcol(runSet->batchCnt*runSet->batchSamplesCnt*runSet->selectedFeaturesCnt, runSet->targetLen, runSet->predictionSFB, 0, runSet->prediction0, true)!=0) return -1;


	//-- feee neurons()
	destroyNeurons();

	return 0;
}
