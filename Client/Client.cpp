#include "..\CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../TimeSerie/TimeSerie.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"


int main() {

	//client3();	
	//client4();
	//client5();
	//client6();
	//client7();
	//client8();
	//client9();
	//client11();
	//if(client13()!=0) printf("CUDA last error=%d\n",cudaGetLastError());
	//system("pause");
	//return -1;

	//--
	tDBConnection* LogDB=new tDBConnection("cuLogUser", "LogPwd", "ALGO");

	tDebugInfo* DebugParms=new tDebugInfo;
	DebugParms->DebugLevel = 0;
	DebugParms->DebugDest = LOG_TO_ORCL;
	DebugParms->DebugDB=LogDB;
	strcpy(DebugParms->fPath, "C:/temp");
	strcpy(DebugParms->fName, "Client.log");
	DebugParms->PauseOnError = 1;
	//--
	DWORD start, end;
	DWORD mainStart=timeGetTime();

	float scaleM, scaleP;

	//-- data params
	int modelFeature[]={ 0,1,2,3};
	int modelFeaturesCnt=sizeof(modelFeature)/sizeof(int);
	int dataTransformation=DT_DELTA;
	int historyLen= 50000;// 140;// 20;// 50000;// 50000;// 20;// 500;
	int sampleLen= 200;// 20; //6;// 200;// 200;
	int predictionLen=2;

	//-- net geometry
	char* levelRatioS="1, 0.5, 1";// "1, 0.5";
	int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
	bool useContext=false;
	bool useBias=false;

	//-- DataSets for train and run. batchSize can be different between the two
	DataSet* trainSet;	int batchsamplesCnt_T=100;// 10;
	DataSet* runSet;	int batchsamplesCnt_R=100;// 10;
	
	//-- logging parameters
	bool saveClient=true;
	bool saveMSE=true;
	bool saveRun=true;
	bool saveW=true;
	bool saveNet=false;

	//-- Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
	NN* trNN=nullptr;
	try {
		trNN=new NN(sampleLen, predictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias);
	}
	catch (const char* e) {
		LogWrite(DebugParms, LOG_ERROR, "NN creation failed. (%s)\n", 1, e);
		return -1;
	}

	//-- 1. load timeserie, transform, and scale it according to level 0 activation
	const int FXfeaturesCnt=5;	//-- OHLC, fixed by the query
	char* tsDate0="201612300000";
	start=timeGetTime();
	TS* ts1=new TS(historyLen, FXfeaturesCnt, DebugParms);
	if (ts1->load(new tFXData("History", "HistoryPwd", "ALGO", "EURUSD", "H1", false), tsDate0)!=0) return -1;
	printf("ts1 create+load, elapsed time=%ld \n", (DWORD)(timeGetTime()-start));	
	//ts1->dump("C:/temp/ts1.orig.csv");

	//-- 2. apply data transformation
	if (ts1->transform(dataTransformation)!=0) return -1;
	//ts1->dump("C:/temp/ts1.tr.csv");

	//-- scale according to activation at network level 0 
	ts1->scale(trNN->scaleMin[0], trNN->scaleMax[0]);
	//ts1->dump("C:/temp/ts1.trs.csv");

	//-- 3. create training dataset from timeserie
	//-- sampleLen/predictionLen is taken from nn
	//-- model features cnt must be taken from nn
	//-- model features list is defined here.
	//-- batch size is defined here, and can be different between train and run datasets

	start=timeGetTime();
	try {
		trainSet=new DataSet(ts1, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_T);
	}
	catch (const char* e) {
		LogWrite(DebugParms, LOG_ERROR, "TRAIN Dataset creation failed: %s (sampleLen=%d, predictionLen=%d, batchSampleCnt=%d)\n", 4, e, sampleLen, predictionLen, batchsamplesCnt_T);
		return -1;
	}
	//trainSet->dump("c:/temp/trainSet.log");
	printf("build train DataSet from ts, elapsed time=%ld \n", (DWORD)(timeGetTime()-start));

	start=timeGetTime();
	try {
		runSet=new DataSet(ts1, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_R);
	}
	catch (const char* e) {
		LogWrite(DebugParms, LOG_ERROR, "RUN Dataset creation failed: %s (sampleLen=%d, predictionLen=%d, batchSampleCnt=%d)\n", 4, e, sampleLen, predictionLen, batchsamplesCnt_R);
		return -1;
	}
	//runSet->dump("C:/temp/runSet.log");
	printf("build run DataSet from ts, elapsed time=%ld \n", (DWORD)(timeGetTime()-start));


	//-- set training parameters
	trNN->MaxEpochs=100;
	trNN->NetSaveFreq=200;
	trNN->TargetMSE=(float)0.0001;
	trNN->BP_Algo=BP_STD;
	trNN->LearningRate=(numtype)0.01;
	trNN->LearningMomentum=(numtype)0.5;
	trNN->StopOnDivergence=true;

	//-- train with training Set, which specifies batch size and features list (not count)
	if(trNN->train(trainSet)!=0) return -1;

	//-- persist MSE 
	if(saveMSE){
		start=timeGetTime();
		if (LogSaveMSE(DebugParms, trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->mseT, trNN->mseV)!=0) return -1;
		printf("LogSaveMSE() elapsed time=%ld \n", (DWORD)(timeGetTime()-start));
	}
	//-- persist final W
	if (saveW) {
		start=timeGetTime();
		if (LogSaveW(DebugParms, trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->weightsCntTotal, trNN->W)!=0) return -1;
		printf("LogSaveW() elapsed time=%ld \n", (DWORD)(timeGetTime()-start));
	}
	//-- Persist network structure
	if (saveNet) {
		//if (LogSaveStruct(DebugParms, trNN->pid, trNN->tid, trNN->InputCount, trNN->OutputCount, trNN->featuresCnt, trNN->sampleLen, trNN->predictionLen, trNN->batchCnt, batchSamplesCount, trNN->useContext, trNN->useBias, trNN->ActivationFunction, trNN->MaxEpochs, trNN->ActualEpochs, trNN->TargetMSE, trNN->StopOnDivergence, trNN->NetSaveFreq, trNN->BP_Algo, trNN->LearningRate, trNN->LearningMomentum)!=0) return -1;
	}

	//-- run
	start=timeGetTime();
	trNN->run(runSet, nullptr);
	printf("run() , elapsed time=%ld \n", (DWORD)(timeGetTime()-start));
	
	//-- persist run
	if (saveRun) {
		start=timeGetTime();
		if (LogSaveRun(DebugParms, trNN->pid, trNN->tid, runSet->samplesCnt, modelFeaturesCnt, runSet->prediction0, runSet->target0)!=0) return -1;
		printf("LogSaveRun(), elapsed time=%ld \n", (DWORD)(timeGetTime()-start));
	}

	//-- destroy training NN
	delete trNN;

	//-- persist client call
	if (saveClient) {
		if (LogSaveClient(DebugParms, GetCurrentProcessId(), "Client.cpp", mainStart, (DWORD)(timeGetTime()-mainStart), 1, tsDate0, 1, 1)!=0) return -1;
	}

	//-- final Commit
	Commit(DebugParms);


	printf("Finished. Total elapsed time=%ld \n", (DWORD)(timeGetTime()-mainStart));

	system("pause");
	return 0;
}
