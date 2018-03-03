#include "..\CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"
#include "../TimeSerie/TimeSerie.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"

int main(int argc, char* argv[]) {

	//-- all variables associated with parameters read from file should be declared here
	int modelSampleLen, modelPredictionLen;
	int modelFeaturesCnt; int* modelFeature;

	//-- training parms
	bool doTrain; bool doTrainRun;
	//-- data source
	int trainTS_DS_type;
	//-- FXDB-type data source properties
	tDBConnection* trainTS_DS_FXDB; char trainTS_DS_FX_DBUser[DBUSER_MAXLEN]; char trainTS_DS_FX_DBPassword[DBPASSWORD_MAXLEN]; char trainTS_DS_FX_DBConnString[DBCONNSTRING_MAXLEN];
	//-- File-type data source properties
	tFileData* trainTS_DS_File; char trainTS_DS_File_FullName[MAX_PATH]; int trainTS_DS_File_FieldSep;
	//-- datasource-independent timeserie properties
	char trainTSdate0[13]; int trainTShistoryLen; int trainTS_DT; 
	//-- dataset properties
	int trainBatchSamplesCnt; int trainSetFeaturesCnt; int* trainSetFeature;

	//-- testing parms
	bool doTest; 
	bool runFromSavedNet; int testWpid; int testWtid;
	//-- data source
	int testTS_DS_type;
	//-- FXDB-type data source properties
	tDBConnection* testTS_DS_FXDB; char testTS_DS_FX_DBUser[DBUSER_MAXLEN]; char testTS_DS_FX_DBPassword[DBPASSWORD_MAXLEN]; char testTS_DS_FX_DBConnString[DBCONNSTRING_MAXLEN];
	//-- File-type data source properties
	tFileData* testTS_DS_File; char testTS_DS_File_FullName[MAX_PATH]; int testTS_DS_File_FieldSep;
	//-- datasource-independent timeserie properties
	char testTSdate0[13]; int testTShistoryLen; int testTS_DT;
	//-- dataset properties
	int testBatchSamplesCnt; int testSetFeaturesCnt; int* testSetFeature;
	
	//-- validation parms
	bool doValid;
	//-- data source
	int validTS_DS_type;
	//-- FXDB-type data source properties
	tDBConnection* validTS_DS_FXDB; char validTS_DS_FX_DBUser[DBUSER_MAXLEN]; char validTS_DS_FX_DBPassword[DBPASSWORD_MAXLEN]; char validTS_DS_FX_DBConnString[DBCONNSTRING_MAXLEN];
	//-- File-type data source properties
	tFileData* validTS_DS_File; char validTS_DS_File_FullName[MAX_PATH]; int validTS_DS_File_FieldSep;
	//-- datasource-independent timeserie properties
	char validTSdate0[13]; int validTShistoryLen; int validTS_DT;
	//-- dataset properties
	int validBatchSamplesCnt; int validSetFeaturesCnt; int* validSetFeature;

	//-- persistor(s) For now, just one for all tables
	int persistorDest;
	char persistorDBUser[30]; char persistorDBPassword[30]; char persistorDBConnString[30];
	tDBConnection* persistorDB;	tLogger* persistor;
	

	//-- start client timer
	DWORD mainStart=timeGetTime();

	//-- main debugger declaration & creation
	createMainDebugger(DBG_LEVEL_ERR, DBG_DEST_BOTH);
	//-- set main debugger properties
	dbg->timing=false;

	//-- everything else must be enclosed in try/catch block
	try {

		//-- create client parms, include command-line parms, and read parameters file
		tParamMgr* parms; safeCallEE(parms=new tParamMgr(new tFileInfo("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.ini", FILE_MODE_READ), argc, argv));

		//-- invariant data model properties
		parms->get(&modelSampleLen, "DataModel.SampleLen");
		parms->get(&modelPredictionLen, "DataModel.PredictionLen");
		parms->get(&modelFeaturesCnt, "DataModel.FeaturesCount");
		
		//-- Train TimeSerie & DataSet
		parms->get(&doTrain, "Train.dTrain");
		if (doTrain) {
			doTrainRun; parms->get(&doTrainRun, "Train.doTrainRun");
			//-- TimeSerie
			parms->get(trainTSdate0, "Train.TimeSerie.Date0");
			parms->get(&trainTShistoryLen, "Train.TimeSerie.HistoryLen");
			parms->get(&trainTS_DT, "Train.TimeSerie.DataTransformation", enumlist);
			parms->get(&trainTS_DS_type, "Train.TimeSerie.DataSource.Type", enumlist);
			if (trainTS_DS_type==SOURCE_DATA_FROM_FXDB) {
				parms->get(trainTS_DS_FX_DBUser, "Train.TimeSerie.DataSource.FXData.DBUser");
				parms->get(trainTS_DS_FX_DBPassword, "Train.TimeSerie.DataSource.FXData.DBPassword");
				parms->get(trainTS_DS_FX_DBConnString, "Train.TimeSerie.DataSource.FXData.DBConnString");
				safeCallEE(trainTS_DS_FXDB=new tDBConnection(trainTS_DS_FX_DBUser, trainTS_DS_FX_DBPassword, trainTS_DS_FX_DBConnString));
			} else if (trainTS_DS_type==SOURCE_DATA_FROM_FILE) {
				parms->get(trainTS_DS_File_FullName, "Train.TimeSerie.DataSource.FileData.FileFullName");
				parms->get(&trainTS_DS_File_FieldSep, "Train.TimeSerie.DataSource.FileData.FieldSep", enumlist);
				safeCallEE(trainTS_DS_File=new tFileData(new tFileInfo(trainTS_DS_File_FullName, FILE_MODE_READ), trainTS_DS_File_FieldSep));
			}
			//-- DataSet
			parms->get(&trainBatchSamplesCnt, "Train.DataSet.BatchSamplesCount");
			int* trainSetFeature; parms->get(&trainSetFeature, "Train.DataSet.SelectedFeatures", false, &trainSetFeaturesCnt);
		}
		//-- Test TimeSerie & DataSet
		parms->get(&doTest, "Test.doTest");
		if (doTest) {
			//-- TimeSerie
			parms->get(testTSdate0, "Test.TimeSerie.Date0");
			parms->get(&testTShistoryLen, "Test.TimeSerie.HistoryLen");
			parms->get(&testTS_DT, "Test.TimeSerie.DataTransformation", enumlist);
			parms->get(&testTS_DS_type, "Test.TimeSerie.DataSource.Type", enumlist);
			if (testTS_DS_type==SOURCE_DATA_FROM_FXDB) {
				parms->get(testTS_DS_FX_DBUser, "Test.TimeSerie.DataSource.FXData.DBUser");
				parms->get(testTS_DS_FX_DBPassword, "Test.TimeSerie.DataSource.FXData.DBPassword");
				parms->get(testTS_DS_FX_DBConnString, "Test.TimeSerie.DataSource.FXData.DBConnString");
				safeCallEE(testTS_DS_FXDB=new tDBConnection(testTS_DS_FX_DBUser, testTS_DS_FX_DBPassword, testTS_DS_FX_DBConnString));
			} else if (testTS_DS_type==SOURCE_DATA_FROM_FILE) {
				parms->get(testTS_DS_File_FullName, "Test.TimeSerie.DataSource.FileData.FileFullName");
				parms->get(&testTS_DS_File_FieldSep, "Test.TimeSerie.DataSource.FileData.FieldSep", enumlist);
				safeCallEE(testTS_DS_File=new tFileData(new tFileInfo(testTS_DS_File_FullName, FILE_MODE_READ), testTS_DS_File_FieldSep));
			}
			//-- DataSet
			parms->get(&testBatchSamplesCnt, "Test.DataSet.BatchSamplesCount");
			int* testSetFeature; parms->get(&testSetFeature, "Test.DataSet.SelectedFeatures", false, &testSetFeaturesCnt);
			//-- load saved Net?
			parms->get(&runFromSavedNet, "Test.RunFromSavedNet");
			if (runFromSavedNet) {
				parms->get(&testWpid, "Test.RunFromSavedNet.ProcessId");
				parms->get(&testWtid, "Test.RunFromSavedNet.ThreadId");
			}
		}
		//-- Validation TimeSerie
		parms->get(&doValid, "Validation.doValidation");
		if(doValid){
			//-- TimeSerie
			parms->get(validTSdate0, "Validation.TimeSerie.Date0");
			validTShistoryLen=trainTShistoryLen;			//-- must be the same (?)
			validTS_DT=trainTS_DT;							//-- must be the same (?)
			validBatchSamplesCnt=trainBatchSamplesCnt;		//-- must be the same (?)
			parms->get(&validTS_DS_type, "Validation.TimeSerie.DataSource.Type", enumlist);
			if (validTS_DS_type==SOURCE_DATA_FROM_FXDB) {
				parms->get(validTS_DS_FX_DBUser, "Validation.TimeSerie.DataSource.FXData.DBUser");
				parms->get(validTS_DS_FX_DBPassword, "Validation.TimeSerie.DataSource.FXData.DBPassword");
				parms->get(validTS_DS_FX_DBConnString, "Validation.TimeSerie.DataSource.FXData.DBConnString");
				safeCallEE(validTS_DS_FXDB=new tDBConnection(validTS_DS_FX_DBUser, validTS_DS_FX_DBPassword, validTS_DS_FX_DBConnString));
			} else if (validTS_DS_type==SOURCE_DATA_FROM_FILE) {
				parms->get(validTS_DS_File_FullName, "Validation.TimeSerie.DataSource.FileData.FileFullName");
				parms->get(&validTS_DS_File_FieldSep, "Validation.TimeSerie.DataSource.FileData.FieldSep", enumlist);
				safeCallEE(validTS_DS_File=new tFileData(new tFileInfo(validTS_DS_File_FullName, FILE_MODE_READ), validTS_DS_File_FieldSep));
			}
		}
		//-- check if model feature selection is coherent among train and test. TO DO!

		//-- create persistor, with its own DBConnection, to save results data.
		parms->get(&persistorDest, "Persistor.Destination", true);
		if (persistorDest==PERSIST_TO_ORCL) {
			parms->get(persistorDBUser, "Persistor.DBUser");
			parms->get(persistorDBPassword, "Persistor.DBPassword");
			parms->get(persistorDBConnString, "Persistor.DBConnString");
			safeCallEE(persistorDB=new tDBConnection(persistorDBUser, persistorDBPassword, persistorDBConnString));
			safeCallEE(persistor=new tLogger(persistorDB));
			parms->get(&persistor->saveNothing, "Persistor.saveNothing");
			parms->get(&persistor->saveClient, "Persistor.saveClient");
			parms->get(&persistor->saveMSE, "Persistor.saveMSE");
			parms->get(&persistor->saveRun, "Persistor.saveRun");
			parms->get(&persistor->saveInternals, "Persistor.saveInternals");
			parms->get(&persistor->saveImage, "Persistor.saveImage");
		} else {
			//-- TO DO ...
		}

		//-- net geometry
		char* levelRatioS= "1,1,1,0.5";//"0.7"
		int activationFunction[]={ NN_ACTIVATION_TANH,NN_ACTIVATION_TANH,NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH, NN_ACTIVATION_TANH };
		bool useContext=false;
		bool useBias=false;

		//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, context, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
		tDbg* NNdbg; safeCallEE(NNdbg=new tDbg(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("NN.log"), true));
		tNN* myNN;   safeCallEE(myNN=new tNN(modelSampleLen, modelPredictionLen, modelFeaturesCnt, levelRatioS, activationFunction, useContext, useBias, NNdbg));
		//-- 0.1. set training parameters
		myNN->MaxEpochs=250;
		myNN->NetSaveFreq=200;
		myNN->TargetMSE=(float)0.0001;
		myNN->BP_Algo=BP_STD;
		myNN->LearningRate=(numtype)0.01;
		myNN->LearningMomentum=(numtype)0.5;
		myNN->StopOnDivergence=true;

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
			tDataSet* trainSet; safeCallEE(trainSet=new tDataSet(trainTS, modelSampleLen, modelPredictionLen, modelFeaturesCnt, modelFeature, trainBatchSamplesCnt));
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
			delete trainSet; delete trainTS; delete trainDataSrc;
		}
		if (doValid) {
			tFXData* validDataSrc; safeCallEE(validDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* validTS; safeCallEE(validTS=new tTS(validDataSrc, validTShistoryLen, validTSdate0, validTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* validSet; safeCallEE(validSet=new tDataSet(validTS, modelSampleLen, modelPredictionLen, modelFeaturesCnt, modelFeature, validBatchSamplesCnt));
			delete validSet; delete validTS; delete validDataSrc;
		}
		if (doTest) {
			tFXData* testDataSrc; safeCallEE(testDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* testTS; safeCallEE(testTS=new tTS(testDataSrc, testTShistoryLen, testTSdate0, testTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* testSet; safeCallEE(testSet=new tDataSet(testTS, modelSampleLen, modelPredictionLen, modelFeaturesCnt, modelFeature, testBatchSamplesCnt));
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
			delete testSet; delete testTS; delete testDataSrc;
		}

		//-- 9. persist Client info
		safeCallEE(persistor->SaveClient(GetCurrentProcessId(), "Client.cpp", mainStart, (DWORD)(timeGetTime()-mainStart), 1, trainTSdate0, doTrain, doTrainRun, doTest));

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
		delete myNN;
		delete parms;

		delete dbg;
	}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed.\n", 0);
		return -1;
	}

	system("pause");
	return 0;

}
