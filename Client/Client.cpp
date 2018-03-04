#include "..\CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"
#include "../TimeSerie/TimeSerie.h"
#include "../SharedUtils/UberSet.h"
#include "..\cuNN\cuNN.h"
#include "../Logger/Logger.h"
#include "../MyAlgebra/MyAlgebra.h"

int main(int argc, char* argv[]) {

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
		tParamMgr* parms; safeCallEE(parms=new tParamMgr(new tFileInfo("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", FILE_MODE_READ), argc, argv));

		//-- Uber-parameters for model (set-invariant) parms
		tUberSetParms* modelParms=new tUberSetParms(parms, US_MODEL, dbg);
		//-- Uber-parameters for train, test, validation sets
		tUberSetParms* trainParms=new tUberSetParms(parms, US_TRAIN, dbg);
		tUberSetParms* testParms =new tUberSetParms(parms, US_TEST , dbg);
		tUberSetParms* validParms=new tUberSetParms(parms, US_VALID, dbg);
		//-- check if model feature selection is coherent among train and test. TO DO!

		//-- create persistor, with its own DBConnection, to save results data.
		parms->get(&persistorDest, "Persistor.Destination", true);
		if (persistorDest==ORCL) {
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

		//-- determine Engine Architecture (number of cores, type and position for every core)

		//-- initialize each core
		
		//-- for now, let's say we have just one core, and it's NN
		tUberSetParms* NNcore0Parms=new tUberSetParms(parms, US_CORE_NN, dbg);

		//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, conTXT, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
		tDbg* NNdbg; safeCallEE(NNdbg=new tDbg(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("NN.log"), true));

		tNN* myNN;   safeCallEE(myNN=new tNN(modelParms->SampleLen, modelParms->PredictionLen, modelParms->FeaturesCnt, NNcore0Parms->NN));
/*
		//-- 1. For each TimeSerie(Training, Validation, Test), do the following:
		//-- 1.1. define its DataSource
		//-- 1.2. call specific constructor to "Prepare" it : Create, LoadData, Transform, Scale
		//-- 1.3. create dataset (1:1 ??)
		//--		sampleLen/predictionLen is taken from nn
		//--		model features cnt must be taken from nn
		//--		model features list is defined here.
		//--		batch size is defined here, and can be different between train and test datasets
		if (doTrain) {
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
*/	}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed.\n", 0);
		return -1;
	}

	system("pause");
	return 0;

}
