#pragma once

#include "..\CommonEnv.h"
#include "../MyUtils/MyUtils.h"
#include "../MyAlgebra/MyAlgebra.h"

//-- Exceptions
#define FAIL_INITCUDA "CUDA Initialization Failed. \n"
#define FAIL_INITCUBLAS "CUBLAS Initialization Failed. \n"
#define FAIL_INITCU "CUDA/CUBLAS Initialization Failed. \n"
#define FAIL_CUDAMALLOC "CUDA malloc failed. \n"
#define FAIL_MALLOC_N "Neurons memory allocation failed. \n"
#define FAIL_MALLOC_W "Weights memory allocation failed. \n"
#define FAIL_FREE_N "Neurons memory free failed. \n"
#define FAIL_FREE_S "Scalar memory free failed. \n"
#define FAIL_FREE_W "Weights memory free failed. \n"
#define FAIL_MALLOC_e "Errors memory allocation failed. \n"
#define FAIL_MALLOC_u "Targets memory allocation failed. \n"
#define FAIL_MALLOC_SCALAR "Scalars memory allocation failed. \n"

#define MLP 0
#define RNN 1
#define MAX_LEVELS 8

//-- Training Protocols
#define TP_STOCHASTIC	0
#define TP_BATCH		1
#define TP_ONLINE		2

//-- Activation Functions
#define NN_ACTIVATION_TANH     1	// y=tanh(x)				=> range is [-1 ÷ 1]
#define NN_ACTIVATION_EXP4     2	// y = 1 / (1+exp(-4*x))	=> range is [ 0 ÷ 1]
#define NN_ACTIVATION_RELU     3	// y=max(0,x)
#define NN_ACTIVATION_SOFTPLUS 4	// y=ln(1+e^x)

//-- Backpropagation algorithms
#define BP_STD			0
#define BP_QING			1
#define BP_RPROP		2
#define BP_QUICKPROP	3
#define BP_SCGD			4 // Scaled Conjugate Gradient Descent
#define BP_LM			5 // Levenberg-Marquardt

typedef struct sNN {
	void* cublasH;
	void* cuRandH;
	void* cuStream[4];

	//-- every instantiation has 1 process id and 1 thread id (TO BE CONFIRMED)
	int pid;
	int tid;

	//-- topology
	int InputCount;
	int OutputCount;
	//--
	int featuresCnt;
	int sampleLen;
	int predictionLen;
	//--
	int batchCnt;
	int batchSamplesCnt;	// usually referred to as Batch Size
	bool useContext;
	bool useBias;

	float levelRatio[MAX_LEVELS];
	int levelsCnt;
	int nodesCnt[MAX_LEVELS];
	int levelFirstNode[MAX_LEVELS];
	int ctxStart[MAX_LEVELS];

	int nodesCntTotal;
	int weightsCnt[MAX_LEVELS-1];
	int weightsCntTotal;
	int levelFirstWeight[MAX_LEVELS-1];

	float scaleMin;
	float scaleMax;

	//-- NNParms
	int ActivationFunction;
	int MaxEpochs;
	float TargetMSE;
	bool StopOnReverse;	// stops training if MSE turns upwards
	int NetSaveFreq;	// saves network weights every <n> epochs
	int BP_Algo;
	float LearningRate;
	float LearningMomentum;

	numtype* a;
	numtype* F;
	numtype* dF;
	numtype* edF;
	numtype* W;
	numtype* dW;
	numtype* dJdW;
	numtype* e;
	numtype* u;
	numtype* TMP;	// used to transpose matrices before multiplication. sized as weightsCnt[0]

	//-- error measuring
	numtype tse;	// total squared error
	numtype se;		// squared sum e
	numtype* mseT;	// Training mean squared error, array indexed by epoch, always on host
	numtype* mseV;	// Validation mean squared error, array indexed by epoch, always on host
	numtype* ss;	// device-side shared scalar value, to be used to calc any of the above
	int ActualEpochs;

	EXPORT sNN(int sampleLen_, int predictionLen_, int featuresCnt_, int batchCnt_, int batchSamplesCnt_, char LevelRatioS_[60], int ActivationFunction_, bool useContext_, bool useBias_);
	EXPORT ~sNN();

	void setLayout(char LevelRatioS[60]);

	EXPORT void setActivationFunction(int func_);
	int sNN::Activate(int level);
	int sNN::calcErr();

	EXPORT int train(numtype* sample, numtype* target);
	EXPORT int run(numtype* runW, int runSampleCnt, numtype* sample, numtype* target, numtype* Oforecast);

} NN;

