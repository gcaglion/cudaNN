#include "..\CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"
#include "../TimeSerie/TimeSerie.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"


int main() {

	//dbgcli();
	//dbgcli2();
	//system("pause");
	//return -1;

	DWORD mainStart=timeGetTime();
	//--
	int clientPid=GetCurrentProcessId();
	int clientTid=GetCurrentThreadId();

	//-- main debugger declaration & creation
	sDbg* dbg=nullptr;
	try {
		dbg=new tDbg(DBG_LEVEL_STD, DBG_DEST_BOTH, new tFileInfo("Client.log", DEBUG_DEFAULT_PATH));
	} catch (std::exception e) {
		if (dbg==nullptr) {
			fprintf(stderr, "\n CRITICAL ERROR: could not create main client debugger!\n");
			system("pause"); return -1;
		}
	}

	//-- everything must be enclosed in try/catch block

	try {

	//-- db connections
	tDBConnection* FXDB=nullptr;
	tDBConnection* persistDB=nullptr;
	//-- results logger
	tLogger* persistor=nullptr;
	//-- results logger's own debugger
	tDbg* persistorDbg=nullptr;
	//-- timeseries & datasets
	tFXData* eurusdH1= nullptr;
	TS* fxTS=nullptr;
	DataSet* trainSet;
	DataSet* runSet;
	//--
	NN* trNN=nullptr;

	float x=12.3456789;
	printf("id=%d; desc=%s;abcd x=%10.4f; int=%d . Bye!\n", 12, "description", x, -3);
	dbg->write(DBG_LEVEL_STD,	"id=%d; desc=%s;abcd x=%10.4f; int=%d . Bye!\n", 4,	12, "description", x, -3);
	system("pause");
		
	dbg->timing=true;
	safeCallEE(FXDB=new tDBConnection("History", "HistoryPwd", "ALGO"));						//-- create DBConnection for FX History DB		
	safeCallEE(eurusdH1=new tFXData(FXDB, "EURUSD", "H1", false));							//-- create FXData for EURUSD H1		
	safeCallEE(persistDB=new tDBConnection("cuLogUser", "LogPwd", "ALGO", persistorDbg));	//-- create DBConnection for Persistor DB
	safeCallEE(persistor=new tLogger(persistDB, persistorDbg));								//-- create Logger from persistDB connection to save results data		
	persistor->saveImage=true;																//-- logger parameters (when different from default settings)

	//-- data params
	int modelFeature[]={ 1 };
	int modelFeaturesCnt=sizeof(modelFeature)/sizeof(int);
	int dataTransformation=DT_DELTA;
	int historyLen= 500;// 50;// 500;// 50000;// 140;// 20;// 50000;// 50000;// 20;// 500;
	int sampleLen=  50;// 3;// 50;//;// 20; //6;// 200;// 200;
	int predictionLen=3;// 1;// 3;

	//-- net geometry
	char* levelRatioS= "1, 0.5";//"1, 0.5, 1";//"0.7"
	int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
	bool useContext=false;
	bool useBias=true;

	//-- DataSets for train and run. batchSize can be different between the two
	int batchsamplesCnt_T=1;// 10;
	int batchsamplesCnt_R=1;// 10;

	//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
	safeCallEE(trNN=new NN(sampleLen, predictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias, new tDbg()));
	//-- 0.1. set training parameters
	trNN->MaxEpochs=50;
	trNN->NetSaveFreq=200;
	trNN->TargetMSE=(float)0.0001;
	trNN->BP_Algo=BP_STD;
	trNN->LearningRate=(numtype)0.01;
	trNN->LearningMomentum=(numtype)0.5;
	trNN->StopOnDivergence=false;

	//-- 1. create timeSerie, set len as the number of time steps, and set featuresCnt based on the expected data it will hold
	const int FXfeaturesCnt=5;	//-- OHLC, fixed by the query
	safeCallEE(fxTS=new TS(historyLen, FXfeaturesCnt));

	//-- 3. load data into fxTS, using FXData info, and start date
	char* tsDate0="201612300000";
	safeCallEE(fxTS->load(eurusdH1, tsDate0));

	//-- 4. apply data transformation
	safeCallEE(fxTS->transform(dataTransformation));

	//-- 5. scale according to activation at network level 0 
	safeCallEE(fxTS->scale(trNN->scaleMin[0], trNN->scaleMax[0]));

	//-- 6. create training dataset from timeserie
	//-- sampleLen/predictionLen is taken from nn
	//-- model features cnt must be taken from nn
	//-- model features list is defined here.
	//-- batch size is defined here, and can be different between train and run datasets

	safeCallEE(trainSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_T));
	safeCallEE(runSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_R));

	//-- 7. train with training Set
	safeCallEE(trNN->train(trainSet));

	//-- 7.1. persist training
	safeCallEE(persistor->SaveMSE(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->mseT, trNN->mseV));
	safeCallEE(persistor->SaveW(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->weightsCntTotal, trNN->W));

	//-- 8. run on the network just trained with runing Set, which specifies batch size and features list (not count)
	safeCallEE(trNN->run(runSet, nullptr));

	//-- 8.1. persist runing
	safeCallEE(persistor->SaveRun(trNN->pid, trNN->tid, runSet->samplesCnt, modelFeaturesCnt, runSet->prediction0, runSet->target0));

	//-- final Commit
	persistor->Commit();

	//-- destroy training NN
	delete trNN;
	} catch(std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "Client failed.", 0);
		return -1;
	}

	system("pause");
	return 0;

}
