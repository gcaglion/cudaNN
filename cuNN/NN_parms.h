#pragma once
#include "../CommonEnv.h"

#define MAX_LEVELS 128

typedef struct sNNparms {

	//-- topology
	int levelsCnt;
	float* levelRatio;
	int* ActivationFunction;	// can be different for each level
	int batchSamplesCnt;	// usually referred to as Batch Size
	Bool useContext;
	Bool useBias;

	//-- training-common
	int MaxEpochs;
	float TargetMSE;
	int NetSaveFreq;
	Bool StopOnDivergence;
	int BP_Algo;
	//-- training-BP_Std specific
	float LearningRate;
	float LearningMomentum;

#ifdef __cplusplus
	sNNparms() {
		levelRatio=(float*)malloc((MAX_LEVELS-2)*sizeof(float));
		ActivationFunction=(int*)malloc(MAX_LEVELS*sizeof(int));
	}
	~sNNparms() {
		free(levelRatio);
		free(ActivationFunction);
	}
#endif

} tNNparms;