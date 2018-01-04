#include "cuNN.h"

sNN::sNN(int sampleLen_, int predictionLen_, int featuresCnt_, int batchCnt_, int batchSamplesCnt_, char LevelRatioS_[60], bool useContext_, bool useBias_) {
	batchCnt=batchCnt_;
	batchSamplesCnt=batchSamplesCnt_;
	useContext=useContext_;
	useBias=useBias_;

	InputCount=sampleLen_*featuresCnt_*batchSamplesCnt;
	OutputCount=predictionLen_*featuresCnt_*batchSamplesCnt;

	//-- 0. init CUDA/BLAS
	cublasH=new void*;
	cuRandH=new void*;
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;

	//-- 1. set Layout
	setLayout(LevelRatioS_);

	//-- 2. malloc neurons and weights on GPU
	if (myMalloc(&N, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&dN, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&edN, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&W, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_e;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_u;

}
sNN::~sNN() {
	//!!!!!!!!!!!!!! create a myFree() functio to handle CUDA-based variables !
	free(weightsCnt);
	free(N);
	free(W);
	//.....
	// free cublasH, cuRandH, curanddestroygenerator...
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
		weightsCnt[l]=nodesCnt[l]*nodesCnt[l+1] /batchSamplesCnt;
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
	int ret, retd;
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		ret=Tanh(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		retd=dTanh(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_EXP4:
		ret=Exp4(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		retd=dExp4(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_RELU:
		ret=Relu(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		retd=dRelu(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_SOFTPLUS:
		ret=SoftPlus(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		retd=dSoftPlus(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	default:
		break;
	}
	return((ret==0&&retd==0)?0:-1);
}

int sNN::train(numtype* sample, numtype* target) {
	int l;
	char fname[MAX_PATH];

	//-- 0. Init
	
	//---- 0.1. Init Neurons (must set context neurons=0, at least for layer 0)
	if( Vinit(nodesCnt[0], &N[0], 0) !=0) return -1;

	//---- 0.2. Init W
	for (l=0; l<(levelsCnt-1); l++) VinitRnd(weightsCnt[l], &W[levelFirstWeight[l]], -1/sqrtf((numtype)nodesCnt[l]), 1/sqrtf((numtype)nodesCnt[l]), cuRandH);
	//dumpData(weightsCntTotal, &W[0], "C:/temp/W.txt");
	//---- 0.3. Init dW
	for (l=0; l<(levelsCnt-1); l++) if( Vinit(weightsCnt[l], &dW[levelFirstWeight[l]], 0) !=0) return -1;

	//-- 1. for every epoch, calc and display MSE
	for(int epoch=0; epoch<MaxEpochs; epoch++) {
		//-- 1.0. train one batch at a time
		for (int b=0; b<batchCnt; b++) {

			//-- 1.0.1.  load samples + targets onto GPU
			if (loadBatchData(&N[0], &sample[b*InputCount], InputCount*sizeof(numtype) )!=0) return -1;
			if (loadBatchData(&u[0], &target[b*OutputCount], OutputCount*sizeof(numtype) )!=0) return -1;
			//dumpData(InputCount, &N[0], "C:/temp/F0.txt");
		
			//-- 1.0.2. reset batch error = 0
			Vinit(nodesCnt[levelsCnt-1], e, 0);
		
			//-- 1.0.3. Feed Forward ( W10[nc1 X nc0] X F0[nc0 X batchSize] => a1 [nc1 X batchSize] )
			for (l=0; l<(levelsCnt-1); l++) {
				int W10y=nodesCnt[l+1]/batchSamplesCnt;
				int W10x=nodesCnt[l]/batchSamplesCnt;
				int W10start= levelFirstWeight[l];
				int N0y=W10x;
				int N0x=batchSamplesCnt;
				int N0start= levelFirstNode[l];
				int N1start= levelFirstNode[l+1];

				//-- N[l+1]=N[l]*W[l]
				if (MbyM(cublasH, W10y, W10x, 1, false, &W[W10start], N0y, N0x, 1, false, &N[N0start], &N[N1start] ) !=0) return -1;
			
				//sprintf(fname, "C:/temp/F%d.txt", l); dumpData(nodesCnt[l], &N[levelFirstNode[l]], fname);
				//sprintf(fname, "C:/temp/F%d.txt", l+1); dumpData(nodesCnt[l+1], &N[levelFirstNode[l+1]], fname);

				//-- l+1 activation
				if(Activate(l+1)!=0) return -1;

			}
		
			//-- 1.0.4. Calc Error
			int outNstart=levelFirstNode[levelsCnt-1];
			if (Vdiff(nodesCnt[levelsCnt-1], &N[outNstart], 1, u, 1, e)!=0) return -1;
			//sprintf(fname, "C:/temp/e.txt"); dumpData(nodesCnt[levelsCnt-1], &N[outNstart], fname);
			//sprintf(fname, "C:/temp/u.txt"); dumpData(nodesCnt[levelsCnt-1], &u[0], fname);

			//-- 1.0.5. BackPropagate, update batch error
			int sc=batchSamplesCnt;
			for (l = levelsCnt-1; l>0; l--) {
				if (l==(levelsCnt-1)) {
					//-- top level only
					VbyV2V(nodesCnt[levelsCnt-1], e, &N[outNstart], &edN[outNstart]);	// edF(l) = e * F'(l)
				} else {
					//-- lower levels
					MbyM(cublasH, sc, nodesCnt[l+1]/sc, 1, false, &edN[levelFirstNode[l+1]], nodesCnt[l+1]/sc, nodesCnt[l]/sc, 1, false, &W[levelFirstWeight[l]], &edN[levelFirstNode[l]]);	// edF(l) = edF(l+1) * WT(l)
					VbyV2V(nodesCnt[l], &edN[levelFirstNode[l]], &dN[levelFirstNode[l]], &edN[levelFirstNode[l]]);	// edF(l) = edF(l) * F'(l)
				}
				//-- common
				MbyM(cublasH, nodesCnt[l+1]/sc, sc, 1, true, &N[levelFirstNode[l-1]], sc, nodesCnt[l]/sc, 1, false, &edN[levelFirstNode[l]], &dJdW[levelFirstNode[l]]);
			}

			//-- 1.0.6. update weights

			//-- dW = LM*dW - LR*dJdW
			if (Vdiff(weightsCntTotal, dW, LearningMomentum, dJdW, LearningRate, dW) !=0) return -1;
			//-- W = W + dW
			if (Vadd(weightsCntTotal, W, 1, dW, 1, W)!=0) return -1;
		}
		//-- 1.1. calc and display MSE
		numtype tse;
		if (Vssum(nodesCnt[levelsCnt-1], e, &tse)!=0) return -1;
		mse=tse/batchCnt/nodesCnt[levelsCnt-1];
		printf("\repoch %d, MSE=%f", epoch, mse);
	}
	//-
	return 0;
}