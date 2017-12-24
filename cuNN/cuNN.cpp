#include "cuNN.h"

void Trim(char* str) {
	int l = 0;
	int i;
	int r = (int)strlen(str);
	char ret[MAX_PATH];
	while (isspace(str[l])>0) l++;
	while (isspace(str[r-1])>0) r--;
	for (i = 0; i<(r-l); i++) ret[i] = str[l+i];
	ret[r-l] = '\0';
	strcpy(str, ret);
}
int cslToArray(char* csl, char Separator, char** StrList) {
	//-- 1. Put a <separator>-separated list of string values into an array of strings, and returns list length
	int i = 0;
	int prevSep = 0;
	int ListLen = 0;
	int kaz;

	while (csl[i]!='\0') {
		kaz = (prevSep==0) ? 0 : 1;
		if (csl[i]==Separator) {
			// separator
			memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
			StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
			Trim(StrList[ListLen]);
			ListLen++;
			prevSep = i;
		}
		i++;
	}
	//-- portion of pDesc after the last comma
	memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
	StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
	Trim(StrList[ListLen]);

	return (ListLen+1);
}

sNN::sNN(int InputCount_, int OutputCount_, int FeaturesCount_, char LevelRatioS_[60], int TotalSamplesCount_, int batchSize_, bool useContext_, bool useGPU) {
	batchSampleCount=batchSize_;
	TotalSamplesCount=TotalSamplesCount_;
	batchCnt=(int)floor(TotalSamplesCount/batchSampleCount);
	useContext=useContext_;

	//-- 0. init CUDA/BLAS
	cublasH=new void*;
	if (myMemInit(cublasH)!=0) throw FAIL_INITCU;

	//-- 1. set Layout
	setLayout(InputCount_, OutputCount_, FeaturesCount_, LevelRatioS_, useContext_);

	//-- 2. malloc neurons and weights on GPU
	if (myMalloc(N, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(dN, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(W, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(e, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_e;


}
sNN::~sNN() {
	free(weightsCnt);
	free(N);
	free(W);
}

void sNN::setLayout(int InputCount_, int OutputCount_, int FeaturesCount_, char LevelRatioS[60], bool useContext_) {
	int i, nl;
	int Levcnt;	// Levels count
	char** DescList=(char**)malloc(MAX_LEVELS*sizeof(char*)); for (i=0; i<MAX_LEVELS; i++) DescList[i]=(char*)malloc(256);

	InputCount=InputCount_; OutputCount=OutputCount_; FeaturesCount=FeaturesCount_; useContext=useContext_;

	//-- 0.1. levels and nodes count
	Levcnt = cslToArray(LevelRatioS, ',', DescList);

	for (i = 0; i<=Levcnt; i++) levelRatio[i] = (numtype)atof(DescList[i]);
	levelsCnt = (Levcnt+2);
	// set nodesCnt (single sample)
	inputSize=InputCount*FeaturesCount;
	outputSize=OutputCount*FeaturesCount;
	nodesCnt[0] = inputSize;
	nodesCnt[levelsCnt-1] = OutputCount;
	for (nl = 0; nl<(levelsCnt-2); nl++) nodesCnt[nl+1] = (int)floor(nodesCnt[nl]*levelRatio[nl]);
	//-- add context neurons
	if (useContext) {
		for (nl = levelsCnt-1; nl>0; nl--) {
			nodesCnt[nl-1] += nodesCnt[nl];
		}
	}
	//-- add one bias neurons for each layer, except output layer
	for (nl = 0; nl<(levelsCnt-1); nl++) nodesCnt[nl] += 1;

	//-- 0.2. weights count
	weightsCntTotal=0;
	for (int l=0; l<(levelsCnt-1); l++) {
		weightsCnt[l]=nodesCnt[l]*nodesCnt[l+1];
		weightsCntTotal+=weightsCnt[l];
	}

	//-- 0.3. multiply nodesCnt[] by batchSize, anc calc nodesCntTotal
	nodesCntTotal=0;
	for (int l=0; l<levelsCnt; l++) {
		nodesCnt[l]*=batchSampleCount;
		nodesCntTotal+=nodesCnt[l];
	}

	//-- 0.4. set first node and first weight for each layer
	for (int l=0; l<levelsCnt; l++) {
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
void sNN::Activate(int level) {
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		Tanh(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		dTanh(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_EXP4:
		Exp4(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		dExp4(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_RELU:
		Relu(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		dRelu(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_SOFTPLUS:
		SoftPlus(nodesCnt[level], &N[levelFirstNode[level]], &N[levelFirstNode[level]]);
		dSoftPlus(nodesCnt[level], &N[levelFirstNode[level]], &dN[levelFirstNode[level]]);
		break;
	default:
		break;
	}
}

/*
int sNN::getLevelFirstNeuron(int l) {
	int ret=0;
	for (int ll=0; ll<l; ll++) ret+=nodesCnt[ll];
	return ret;
}
int sNN::getLevelFirstWeight(int l) {
	int ret=0;
	for (int ll=0; ll<l; ll++) ret+=weightsCnt[ll];
	return ret;
}
*/
int sNN::train(numtype* sample, numtype* target) {
	int l;

	//-- 0. Init
	//---- 0.1. Init Neurons
	//---- 0.2. Init Weights

	//-- 1. train one batch at a time
	int batchMemInputSize=inputSize*batchSampleCount*sizeof(numtype);
	int batchMemOutputSize=outputSize*batchSampleCount*sizeof(numtype);
	for (int b=0; b<batchCnt; b++) {
		//-- 1.1.  load samples + targets onto GPU
		if (loadBatchData(&N[0], &sample[b*batchMemInputSize], batchMemInputSize)!=0) return -1;
		if (loadBatchData(&u[0], &target[b*batchMemOutputSize], batchMemOutputSize)!=0) return -1;
		//-- 1.2. reset batch error = 0
		Vinit(nodesCnt[levelsCnt-1], e, 0);
		//-- 1.3. Feed Forward ( W10[nc1 X nc0] X F0[nc0 X batchSize] => a1 [nc1 X batchSize] )
		for (l=0; l<(levelsCnt-1); l++) {
			int W10start= levelFirstWeight[l];
			int N0start= levelFirstNode[l];
			int N1start= levelFirstNode[l+1];

			//-- N[l+1]=N[l]*W[l]
			if (MbyM(nodesCnt[l+1], nodesCnt[l], 1, &W[W10start], nodesCnt[l], batchSampleCount, 1, &N[N0start], &N[N1start] ) !=0) return -1;
			//-- activation
			Activate(l);

		}
		//-- 1.4. Calc Error
		int outNstart=levelFirstNode[levelsCnt-1];
		if (Vdiff(nodesCnt[levelsCnt-1], &N[outNstart], u, e)!=0) return -1;
		//-- 1.5. BackPropagate, update batch error
/*		for (l = levelsCnt-1; l>0; l--) {
			if (l==(levelsCnt-1)) {
				//-- top level only
				MbyM()
				VbyV2V(NNParms->NodesCount[TOTNODE][l], Mx->NN.e[t0], Mx->NN.dF[l][t0], Mx->NN.edF[l][t0]);													// edF(l) = e * F'(l)
			} else {
				//-- lower levels
				MbyV(NNParms->NodesCount[TOTNODE][l+1], NNParms->NodesCount[TOTNODE][l], Mx->NN.W[l][t0], true, Mx->NN.edF[l+1][t0], Mx->NN.edF[l][t0]);	// edF(l) = edF(l+1) * WT(l+1)
				VbyV2V(NNParms->NodesCount[TOTNODE][l], Mx->NN.edF[l][t0], Mx->NN.dF[l][t0], Mx->NN.edF[l][t0]);											// edF(l) = edF(l)   * F'(l)
			}
			//-- common
			VbyV2M(NNParms->NodesCount[TOTNODE][l], Mx->NN.edF[l][t0], NNParms->NodesCount[TOTNODE][l-1], Mx->NN.F[l-1][t0], false, Mx->NN.dJdW[l-1][t0]);	// dJdW(l) = edF(l) * F(l-1)
		}
*/
		//-- 1.6. update weights
	}
	//-
	return 0;
}