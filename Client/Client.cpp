#include "..\CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../TimeSerie/TimeSerie.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"

void dbgcli() {
	try {
		tDebugInfo* dbg1=new tDebugInfo();
	}
	catch (std::exception e) {
		printf("%s\n", e.what());
	}
}

int main() {

	//dbgcli();
	//system("pause");
	//return -1;

	DWORD mainStart=timeGetTime();
	//--
	int clientPid=GetCurrentProcessId();
	int clientTid=GetCurrentThreadId();

	//-- main objects
	tDBConnection* FXDB=nullptr;
	tDBConnection* persistDB=nullptr;
	tLogger* persistor=nullptr;
	NN* trNN=nullptr;
	tFXData* eurusdH1= nullptr;
	TS* fxTS=nullptr;

	//-- debugger settings
	tDebugInfo* clientDbg=new tDebugInfo(DBG_LEVEL_STD, DBG_DEST_BOTH, new tFileInfo("Client.log", DEBUG_DEFAULT_PATH));
	tDebugInfo* persistorDbg=new tDebugInfo(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("Persistor.log", DEBUG_DEFAULT_PATH));
	//--

	safeCallE(clientDbg, "create DBConnection for FX History DB", FXDB=new tDBConnection("History", "HistoryPwd", "ALGO"));
	safeCallE(clientDbg, "create FXData for EURUSD H1", eurusdH1=new tFXData(FXDB, "EURUSD", "H1", false));
	safeCallE(persistorDbg, "create DBConnection for Persistor DB", persistDB=new tDBConnection("cuLogUser", "LogPwd", "ALGO"));
	safeCallE(persistorDbg, "create Logger from persistDB connection to save results data", persistor=new tLogger(persistorDbg, persistDB));
	//-- logger parameters (when different from default settings
	persistor->saveImage=true;

	//-- data params
	int modelFeature[]={ 1};
	int modelFeaturesCnt=sizeof(modelFeature)/sizeof(int);
	int dataTransformation=DT_DELTA;
	int historyLen= 500;// 50;// 500;// 50000;// 140;// 20;// 50000;// 50000;// 20;// 500;
	int sampleLen=  50;// 3;// 50;//;// 20; //6;// 200;// 200;
	int predictionLen=3;// 1;// 3;

	//-- net geometry
	//char* levelRatioS= "0.7";
	char* levelRatioS= "1, 0.5";//"1, 0.5, 1";//
	int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
	bool useContext=false;
	bool useBias=true;

	//-- DataSets for train and run. batchSize can be different between the two
	DataSet* trainSet;	int batchsamplesCnt_T=1;// 10;
	DataSet* runSet;	int batchsamplesCnt_R=1;// 10;


	//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
	safeCallE(clientDbg, "NN creation", trNN=new NN(sampleLen, predictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias) );
	//-- set training parameters
	trNN->MaxEpochs=50;
	trNN->NetSaveFreq=200;
	trNN->TargetMSE=(float)0.0001;
	trNN->BP_Algo=BP_STD;
	trNN->LearningRate=(numtype)0.01;
	trNN->LearningMomentum=(numtype)0.5;
	trNN->StopOnDivergence=false;

	//-- 1. create timeSerie, set len as the number of time steps, and set featuresCnt based on the expected data it will hold
	const int FXfeaturesCnt=5;	//-- OHLC, fixed by the query
	safeCallE(clientDbg, "FX TimeSerie creation", fxTS=new TS(historyLen, FXfeaturesCnt); );

	//-- 3. load data into fxTS, using FXData info, and start date
	char* tsDate0="201612300000";
	safeCallR(clientDbg, "TS load of FXData", fxTS->load(eurusdH1, tsDate0) );

	//-- 4. apply data transformation
	safeCallR(clientDbg, "apply data transformation", fxTS->transform(dataTransformation) );

	//-- 5. scale according to activation at network level 0 
	safeCallR(clientDbg, "scale data according to L0 activation", fxTS->scale(trNN->scaleMin[0], trNN->scaleMax[0]) );

	//-- 6. create training dataset from timeserie
	//-- sampleLen/predictionLen is taken from nn
	//-- model features cnt must be taken from nn
	//-- model features list is defined here.
	//-- batch size is defined here, and can be different between train and run datasets
	safeCallE(clientDbg, "create train dataset from timeserie", trainSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_T));
	safeCallE(clientDbg, "create run   dataset from timeserie", runSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_R));

	//-- 7. train with training Set

/*	#define block trNN->train(trainSet)
	#define DBG clientDbg
	try { block; }
	catch (std::exception e) {
			sprintf_s(DBG->stackmsg, "%s\n%s(): Error %d at line %d", DBG->stackmsg, __func__, errno, __LINE__);
			DBG->write(DBG_LEVEL_ERR, "%s failed. Exception %s \n", 2, DBG->stackmsg, e.what());
			return -1;
	}
*/
	safeCallTop(clientDbg, "train with train Set", trNN->train(trainSet));

	//-- 7.1. persist training
	safeCallR(persistorDbg, "persist MSE", persistor->SaveMSE(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->mseT, trNN->mseV));
	safeCallR(persistorDbg, "persist final W", persistor->SaveW(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->weightsCntTotal, trNN->W));

	//-- 8. run on the network just trained with runing Set, which specifies batch size and features list (not count)
	safeCallR(clientDbg, "run with run Set", trNN->run(runSet, nullptr));

	//-- 8.1. persist runing
	safeCallR(persistorDbg, "persist run", persistor->SaveRun(trNN->pid, trNN->tid, runSet->samplesCnt, modelFeaturesCnt, runSet->prediction0, runSet->target0));

	//-- final Commit
	persistor->Commit();

	//-- destroy training NN
	delete trNN;

	printf("Finished. Total elapsed time=%ld \n", (DWORD)(timeGetTime()-mainStart));

	system("pause");
	return 0;

}
