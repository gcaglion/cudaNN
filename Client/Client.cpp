#include "..\CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"
#include "../TimeSerie/TimeSerie.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"

int main(int argc, char* argv[]) {

	DWORD mainStart=timeGetTime();
	//--
	int clientPid=GetCurrentProcessId();
	int clientTid=GetCurrentThreadId();

	//-- main debugger declaration & creation
	createMainDebugger(DBG_LEVEL_STD, DBG_DEST_BOTH);
	//-- set main debugger properties
	dbg->timing=false;

	//-- everything else must be enclosed in try/catch block
	try {

		//-- create client parms, include command-line parms, and read parameters file
		tParamMgr* parms; safeCallEE(parms=new tParamMgr(new tFileInfo("C:\\Users\\giacomo.caglioni\\dev\\cudaNN\\Client\\Client.ini", FILE_MODE_READ), argc, argv));

		//-- invariant data shape
		int sampleLen;		parms->get(&sampleLen, "DataParms.SampleLen");
		int predictionLen;	parms->get(&predictionLen, "DataParms.PredictionLen");

/*		float targetMSE;	parms->get(&targetMSE, "NNInfo.TargetMSE");
		char dbuser[30];	parms->get(dbuser, "DataSource.DBConn.DBUser");
		int EngineType;		parms->get(&EngineType, "Forecaster.Engine", true);
		int* fileDS=(int*)malloc(MAXDATASETS*sizeof(int));
		parms->get(&fileDS, "DataSource.FileDatasets");
		float* lr=(float*)malloc(8*sizeof(float));
		parms->get(&lr, "NNInfo.LevelRatios");
		char** tsf=(char**)malloc(12*sizeof(char*)); for (int i=0; i<12; i++) tsf[i]=(char*)malloc(16);
		parms->get(tsf, "DataParms.TSFeatures");
		int* tsfid=(int*)malloc(12*sizeof(int));
		parms->get(&tsfid, "DataParms.TSFeatures", enumlist);
*/

		//-- TRAIN timeseries & datasets
		bool doTrain=true;
		bool doTrainRun =true;				//-- In-Sample test. Runs on Training set.
		char* trainTSdate0="201712300000";
		int trainTShistoryLen=503;
		int trainTS_DT=DT_DELTA;
		int batchSamplesCnt_Train=10;

		//-- TEST timeseries & datasets
		bool doTestRun  =true;				//-- Out-of-Sample test. Runs on Test set.
		//int testWpid=76168, testWtid=77828;			//-- if both 0, use currently loaded W
		int testWpid=0, testWtid=0;			//-- if both 0, use currently loaded W
		char* testTSdate0="201612300000";
		int testTShistoryLen=503;			//-- can be different
		int testTS_DT=DT_DELTA;				//-- can be different
		int batchSamplesCnt_Test=10;			//-- can be different
		
		//-- VALIDATION timeseries & datasets
		bool doValid=false;
		char* validTSdate0="201412300000";
		int validTShistoryLen=trainTShistoryLen;			//-- must be the same (?)
		int validTS_DT=trainTS_DT;							//-- must be the same (?)
		int batchSamplesCnt_Valid=batchSamplesCnt_Train;	//-- must be the same (?)

		//-- NN
		tNN* myNN=nullptr;
		//-- NN own debugger
		tDbg* NNdbg=nullptr;

		//-- create additional debuggers
		safeCallEE(tDbg* NNdbg=new tDbg(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("NN.log"), true));

		//-- create persistor, with its own DBConnection, to save results data. In this case, we want it to have its own debugger
		tDbg*			persistorDbg; safeCallEE(persistorDbg=new tDbg(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("persistor.log"), true));
		tDBConnection*	persistorDB;  safeCallEE(persistorDB=new tDBConnection("cuLogUser", "LogPwd", "ALGO", persistorDbg));
		tLogger*		persistor;	  safeCallEE(persistor=new tLogger(persistorDB, persistorDbg));
		//-- logger parameters (when different from default settings)
		persistor->saveImage=true;

		//-- create DBConnection for FX History DB (common to all TimeSeries)
		tDBConnection* FXDB; safeCallEE(FXDB=new tDBConnection("History", "HistoryPwd", "ALGO"));

		//-- data params
		int modelFeature[]={ 0,1,2,3 };	//-- features are inserted in Dataset in ascending order, regardless of the order specified here. Should be okay...
		int modelFeaturesCnt=sizeof(modelFeature)/sizeof(int);

		//-- net geometry
		char* levelRatioS= "0.5";//"0.7"
		int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
		bool useContext=true;
		bool useBias=false;

		//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
		safeCallEE(myNN=new tNN(sampleLen, predictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias, NNdbg));
		//-- 0.1. set training parameters
		myNN->MaxEpochs=50;
		myNN->NetSaveFreq=200;
		myNN->TargetMSE=(float)0.0001;
		myNN->BP_Algo=BP_STD;

		myNN->LearningRate=(numtype)0.005;
		myNN->LearningMomentum=(numtype)0.5;
		myNN->StopOnDivergence=false;

		//-- 1. For each TimeSerie(Training, Validation, Test), do the following:
		//-- 1.1. define its DataSource
		//-- 1.2. call specific constructor to "Prepare" it : Create, LoadData, Transform, Scale
		//-- 1.3. create dataset (1:1 ??)
		//--		sampleLen/predictionLen is taken from nn
		//--		model features cnt must be taken from nn
		//--		model features list is defined here.
		//--		batch size is defined here, and can be different between train and test datasets
		if (doTrain) {
			tFXData* trainDataSrc; safeCallEE(trainDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* trainTS; safeCallEE(trainTS=new tTS(trainDataSrc, trainTShistoryLen, trainTSdate0, trainTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* trainSet; safeCallEE(trainSet=new tDataSet(trainTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchSamplesCnt_Train));
			//-- train with training set
			safeCallEE(myNN->train(trainSet));
			//-- persist training (MSE + W). Whether or not this gets actually done is controlled by SLogger properties
			safeCallEE(persistor->SaveMSE(myNN->pid, myNN->tid, myNN->ActualEpochs, myNN->mseT, myNN->mseV));
			safeCallEE(persistor->SaveW(myNN->pid, myNN->tid, myNN->ActualEpochs, myNN->weightsCntTotal, myNN->W));
			//-- inference from training set (run + persist)
			if (doTrainRun) {
				safeCallEE(myNN->run(trainSet));
				safeCallEE(persistor->SaveRun(myNN->pid, myNN->tid, 0, myNN->pid, myNN->tid, trainSet->samplesCnt, modelFeaturesCnt, modelFeature, trainSet->prediction0, trainSet->target0));
			}
		}
		if (doValid) {
			tFXData* validDataSrc; safeCallEE(validDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* validTS; safeCallEE(validTS=new tTS(validDataSrc, validTShistoryLen, validTSdate0, validTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* validSet; safeCallEE(validSet=new tDataSet(validTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchSamplesCnt_Valid));
		}
		if (doTestRun) {
			tFXData* testDataSrc; safeCallEE(testDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* testTS; safeCallEE(testTS=new tTS(testDataSrc, testTShistoryLen, testTSdate0, testTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* testSet; safeCallEE(testSet=new tDataSet(testTS, sampleLen, predictionLen, modelFeaturesCnt, modelFeature, batchSamplesCnt_Test));
			if (testWpid!=0&&testWtid!=0) {
				//-- if we specifed both testWpid and testWtid, then load training Weights('-1' stands for 'latest epoch')
				safeCallEE(persistor->LoadW(testWpid, testWtid, -1, myNN->weightsCntTotal, myNN->W));
			} else {
				//-- otherwise, testW pid/tid are those from current training session; also, use existing W from training
				testWpid=myNN->pid; testWtid=myNN->tid;
			}
			//-- Inference from Test Set (run + persist)
			safeCallEE(myNN->run(testSet));
			safeCallEE(persistor->SaveRun(myNN->pid, myNN->tid, 1, testWpid, testWtid, testSet->samplesCnt, modelFeaturesCnt, modelFeature, testSet->prediction0, testSet->target0));
		}

		//-- 9. persist Client info
		safeCallEE(persistor->SaveClient(GetCurrentProcessId(), "Client.cpp", mainStart, (DWORD)(timeGetTime()-mainStart), 1, trainTSdate0, doTrain, doTrainRun, doTestRun));

		dbg->write(DBG_LEVEL_STD, "\nTotal Client Elapsed time: %.4f s.\n", 1, ((timeGetTime()-mainStart)/(float)1000));
		//-- final Commit
		printf("Confirm Final Commit? (Y/N)");
		while (true) {
			int c=getchar();
			if (c=='y'||c=='Y') {
				persistor->Commit();
				break;
			}
			if (c=='n'||c=='N') {
				break;
			}
		}
		

		//-- destroy all objects (therefore all tDbg* objects, therefore all empty debug files)
		delete FXDB;
		delete persistor;
/*		delete trainDataSrc;
		delete trainTS;
		if(trainSet!=nullptr) delete trainSet;
		if(testSet!=nullptr) delete testSet;
*/		delete myNN;

		delete dbg;
	}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed.\n", 0);
		return -1;
	}

	system("pause");
	return 0;

}
