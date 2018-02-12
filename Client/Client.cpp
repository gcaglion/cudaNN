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
	tDebugInfo* clientDbg=new tDebugInfo(1, "Client.log", DEBUG_DEFAULT_PATH, true);
	clientDbg->PauseOnError = 1;
	tDebugInfo* persistorDbg=new tDebugInfo(2, "Persistor.log", DEBUG_DEFAULT_PATH);
	//--

	safeCallE("create DBConnection for FX History DB", clientDbg, FXDB=new tDBConnection("History", "HistoryPwd", "ALGO"));
	safeCallE("create FXData for EURUSD H1", clientDbg, eurusdH1=new tFXData(FXDB, "EURUSD", "H1", false));
	safeCallE("create DBConnection for Logger DB", clientDbg, persistDB=new tDBConnection("cuLogUser", "LogPwd", "ALGO"));
	safeCallE("create Logger from persistDB connection to save results data", clientDbg, persistor=new tLogger(persistorDbg, persistDB));
	//-- logger parameters (when different from default settings
	persistor->saveImage=true;

	//-- data params
	int modelFeature[]={ 1};
	int modelFeaturesCnt=sizeof(modelFeature)/sizeof(int);
	int dataTransformation=DT_DELTA;
	int historyLen= 500;// 50000;// 140;// 20;// 50000;// 50000;// 20;// 500;
	int sampleLen= 50;//;// 20; //6;// 200;// 200;
	int predictionLen=2;

	//-- net geometry
	char* levelRatioS= "1, 0.5";//"1, 0.5, 1";//
	int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
	bool useContext=false;
	bool useBias=true;

	//-- DataSets for train and run. batchSize can be different between the two
	DataSet* trainSet;	int batchsamplesCnt_T=1;// 10;
	DataSet* runSet;	int batchsamplesCnt_R=1;// 10;


	//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
	safeCallE("NN creation", clientDbg, { trNN=new NN(sampleLen, predictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias); });
	//-- set training parameters
	trNN->MaxEpochs=300;
	trNN->NetSaveFreq=200;
	trNN->TargetMSE=(float)0.0001;
	trNN->BP_Algo=BP_STD;
	trNN->LearningRate=(numtype)0.01;
	trNN->LearningMomentum=(numtype)0.5;
	trNN->StopOnDivergence=false;

	//-- 1. create timeSerie, set len as the number of time steps, and set featuresCnt based on the expected data it will hold
	const int FXfeaturesCnt=5;	//-- OHLC, fixed by the query
	safeCallE("FX TimeSerie creation", clientDbg, fxTS=new TS(historyLen, FXfeaturesCnt); );

	//-- 3. load data into fxTS, using FXData info, and start date
	char* tsDate0="201612300000";
	safeCallR("TS load of FXData", clientDbg, fxTS->load(eurusdH1, tsDate0) );

	//-- 4. apply data transformation
	safeCallR("apply data transformation", clientDbg, fxTS->transform(dataTransformation) );

	//-- 5. scale according to activation at network level 0 
	safeCallR("scale data according to L0 activation", clientDbg, fxTS->scale(trNN->scaleMin[0], trNN->scaleMax[0]) );

	//-- 6. create training dataset from timeserie
	//-- sampleLen/predictionLen is taken from nn
	//-- model features cnt must be taken from nn
	//-- model features list is defined here.
	//-- batch size is defined here, and can be different between train and run datasets
	safeCallE("create train dataset from timeserie", clientDbg, trainSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_T));
	safeCallE("create run   dataset from timeserie", clientDbg,   runSet=new DataSet(fxTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchsamplesCnt_R));

	//-- 7. train with training Set, which specifies batch size and features list (not count)
	safeCallR("train with train Set", clientDbg, trNN->train(trainSet));

	//-- 7.1. persist training
	safeCallR("persist MSE", clientDbg, persistor->SaveMSE(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->mseT, trNN->mseV));
	safeCallR("persist final W", clientDbg, persistor->SaveW(trNN->pid, trNN->tid, trNN->ActualEpochs, trNN->weightsCntTotal, trNN->W)); 

	//-- 8. run on the network just trained with runing Set, which specifies batch size and features list (not count)
	safeCallR("run with run Set", clientDbg, trNN->run(runSet, nullptr));

	//-- 8.1. persist runing
	safeCallR("persist run", clientDbg, persistor->SaveRun(trNN->pid, trNN->tid, runSet->samplesCnt, modelFeaturesCnt, runSet->prediction0, runSet->target0));

	//-- final Commit
	persistor->Commit();

	//-- destroy training NN
	delete trNN;

	printf("Finished. Total elapsed time=%ld \n", (DWORD)(timeGetTime()-mainStart));

	system("pause");
	return 0;
}
