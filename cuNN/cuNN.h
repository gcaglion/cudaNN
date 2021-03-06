#pragma once

#include "..\CommonEnv.h"
#include "../MyAlgebra/MyAlgebra.h"
#include "../TimeSerie/TimeSerie.h"

//-- Training Protocols
#define TP_STOCHASTIC	0
#define TP_BATCH		1
#define TP_ONLINE		2

//-- Activation Functions
#define NN_ACTIVATION_TANH     1	// y=tanh(x)				=> range is [-1 � 1]
#define NN_ACTIVATION_EXP4     2	// y = 1 / (1+exp(-4*x))	=> range is [ 0 � 1]
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

	tDbg* dbg;

	//-- MyAlgebra common structures
	Algebra* Alg;

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

	float* levelRatio;
	int levelsCnt;
	int* nodesCnt;
	int* levelFirstNode;
	int* ctxStart;

	int nodesCntTotal;
	int* weightsCnt;
	int weightsCntTotal;
	int* levelFirstWeight;

	//-- set at each level according to ActivationFunction
	float* scaleMin;	
	float* scaleMax;

	//-- NNParms
	int* ActivationFunction;	// can be different for each level
	int MaxEpochs;
	int ActualEpochs;
	float TargetMSE;
	bool StopOnDivergence;	// stops training if MSE turns upwards
	int NetSaveFreq;	// saves network weights every <n> epochs
	int BP_Algo;
	float LearningRate;
	float LearningMomentum;

	numtype* a;
	numtype* F;
	numtype* dF;
	numtype* edF;
	numtype* W;
	numtype* prevW;
	numtype* dW;
	numtype* dJdW;
	numtype* e;
	numtype* u;

	//-- error measuring
	numtype* tse;	// total squared error.	Scalar. On GPU (if used)
	numtype* se;	// squared sum error.	Scalar. On GPU (if used)
	//--
	numtype* mseT;	// Training mean squared error, array indexed by epoch, always on host
	numtype* mseV;	// Validation mean squared error, array indexed by epoch, always on host
	//--

	//-- performance counters
	DWORD LDstart, LDtimeTot=0, LDcnt=0; float LDtimeAvg;
	DWORD FFstart, FFtimeTot=0, FFcnt=0; float FFtimeAvg;
	DWORD FF0start, FF0timeTot=0, FF0cnt=0; float FF0timeAvg;
	DWORD FF1start, FF1timeTot=0, FF1cnt=0; float FF1timeAvg;
	DWORD FF1astart, FF1atimeTot=0, FF1acnt=0; float FF1atimeAvg;
	DWORD FF1bstart, FF1btimeTot=0, FF1bcnt=0; float FF1btimeAvg;
	DWORD FF2start, FF2timeTot=0, FF2cnt=0; float FF2timeAvg;
	DWORD CEstart, CEtimeTot=0, CEcnt=0; float CEtimeAvg;
	DWORD VDstart, VDtimeTot=0, VDcnt=0; float VDtimeAvg;
	DWORD VSstart, VStimeTot=0, VScnt=0; float VStimeAvg;
	DWORD BPstart, BPtimeTot=0, BPcnt=0; float BPtimeAvg;
	DWORD WUstart, WUtimeTot=0, WUcnt=0; float WUtimeAvg;
	DWORD TRstart, TRtimeTot=0, TRcnt=0; float TRtimeAvg;

	EXPORT sNN(int sampleLen_, int predictionLen_, int featuresCnt_, char LevelRatioS_[60], int* ActivationFunction, bool useContext_, bool useBias_, tDbg* dbg_=nullptr);
	EXPORT ~sNN();

	void setLayout(char LevelRatioS_[60], int batchSamplesCnt_);

	EXPORT void setActivationFunction(int* func_);
	void FF();
	void Activate(int level);
	void calcErr();
	void ForwardPass(tDataSet* ds, int batchId, bool haveTargets);
	bool epochMetCriteria(int epoch, DWORD starttime, bool displayProgress=true);
	void BP_std();
	void WU_std();
	void BackwardPass(tDataSet* ds, int batchId, bool updateWeights);

	EXPORT void train(tDataSet* trainSet);
	EXPORT void run(tDataSet* runSet);

private:
	//-- malloc + init
	void mallocNeurons();
	void initNeurons();
	void createWeights();
	//-- free
	void destroyNeurons();
	void destroyWeights();

} tNN;

