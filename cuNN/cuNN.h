#pragma once

#include "..\CommonEnv.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"
#include "../MyAlgebra/MyAlgebra.h"
#include "../TimeSerie/DataSet.h"
#include "../Core/Core.h"
#include "NN_parms.h"
#include "NN_enums.h"

typedef struct sNN :public sCore, public sBaseObj {

	tDebugger* dbg=nullptr;

	//-- MyAlgebra common structures
	Algebra* Alg=nullptr;

	//-- every instantiation has 1 process id and 1 thread id (TO BE CONFIRMED)
	int pid;
	int tid;

	//-- topology
	//int layout->inputCnt;
	//int layout->outputCnt;
	//--
	int featuresCnt;
	int sampleLen;
	int predictionLen;
	//--
	int batchCnt;

	int outputLevel;
	int* nodesCnt;
	int* levelFirstNode;
	int* ctxStart;
	int ActualEpochs;

	int nodesCntTotal;
	int* weightsCnt;
	int weightsCntTotal;
	int* levelFirstWeight;

	//-- set at each level according to ActivationFunction
	float* scaleMin;	
	float* scaleMax;

	//-- NNParms
	tNNparms* parms;

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

	EXPORT sNN(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT sNN(int sampleLen_, int predictionLen_, int featuresCnt_, tNNparms* NNparms_, tDebugger* dbg_=nullptr);
	EXPORT ~sNN();

	void setLayout(int batchSamplesCnt_);

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

