#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../Forecaster/Forecaster.h"

int main(int argc, char* argv[]) {

	//enumcli();
	//system("pause");
	//return -1;

	//-- start client timer
	DWORD mainStart=timeGetTime();

	//-- main debugger declaration & creation
	createMainDebugger(DBG_LEVEL_STD, DBG_DEST_BOTH);
	//-- set main debugger properties
	dbg->timing=false;

	//-- everything else must be enclosed in try/catch block
	try {

		//-- create client parms, include command-line parms, and read parameters file
		tParmsSource* XMLparms; safeCallEE(XMLparms=new tParmsSource("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv));

		XMLparms->parse();

		//-- create Data Forecaster from parms
		tForecaster* forecaster; safeCallEE(forecaster=new tForecaster(XMLparms, "Forecaster", dbg));
/*
		//-- create TimeSeries and DataSets from parms (Train, Test, Validation)
		tTimeSerie* trainTS; tDataSet* trainDS;
		if (forecaster->doTrain) {
			safeCallEE(trainTS=new tTimeSerie(XMLparms, TRAIN_SET, dbg));
			safeCallEE(trainDS=new tDataSet  (XMLparms, trainTS, dbg));
		}
		tTimeSerie* testTS; tDataSet* testDS;
		if (forecaster->doTest) {
			safeCallEE(testTS=new tTimeSerie(XMLparms, TEST_SET, dbg));
			safeCallEE(testDS=new tDataSet(XMLparms, testTS, dbg));
		}
		tTimeSerie* validTS; tDataSet* validDS;
		if (forecaster->doValidation) {
			safeCallEE(validTS=new tTimeSerie(XMLparms, VALID_SET, dbg));
			safeCallEE(validDS=new tDataSet(XMLparms, validTS, dbg));
		}

		//-- create single persistor, with its own DBConnection, to save results data.
		tLogger* persistor; safeCallEE(persistor=new tLogger(XMLparms, ".Forecaster.Persistor", dbg));

		//-- create Engine from parms
		tEngine* engine=new tEngine(XMLparms, ".Forecaster.Engine", dbg);
*/
/*
		//-- initialize each core
		
		//-- for now, let's say we have just one core, and it's NN
		tUberSetParms* NNcore0Parms=new tUberSetParms(XMLparms, US_CORE_NN, dbg);

		//-- 0. Create network based only on sampleLen, predictionLen, geometry (level ratios, conTXT, bias). This sets scaleMin[] and ScaleMax[] needed to proceed with datasets
		tDebugger* NNdbg; safeCallEE(NNdbg=new tDebugger(DBG_LEVEL_ERR, DBG_DEST_BOTH, new tFileInfo("NN.log"), true));

		tNN* myNN;   safeCallEE(myNN=new tNN(forecasterXMLparms->SampleLen, forecasterXMLparms->PredictionLen, forecasterXMLparms->FeaturesCnt, NNcore0XMLparms->NN));

		//-- 1. For each TimeSerie(Training, Validation, Test), do the following:
		//-- 1.1. define its DataSource
		//-- 1.2. call specific constructor to "Prepare" it : Create, LoadData, Transform, Scale
		//-- 1.3. create dataset (1:1 ??)
		//--		sampleLen/predictionLen is taken from nn
		//--		forecaster features cnt must be taken from nn
		//--		forecaster features list is defined here.
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
				safeCallEE(persistor->SaveRun(myNN->pid, myNN->tid, 0, myNN->pid, myNN->tid, trainSet->samplesCnt, forecasterFeaturesCnt, forecasterFeature, trainSet->prediction0, trainSet->target0));
			}
			delete trainSet; delete trainTS; delete trainDataSrc;
		}
		if (doValid) {
			tFXData* validDataSrc; safeCallEE(validDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* validTS; safeCallEE(validTS=new tTS(validDataSrc, validTShistoryLen, validTSdate0, validTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* validSet; safeCallEE(validSet=new tDataSet(validTS, forecasterSampleLen, forecasterPredictionLen, forecasterFeaturesCnt, forecasterFeature, validBatchSamplesCnt));
			delete validSet; delete validTS; delete validDataSrc;
		}
		if (doTest) {
			tFXData* testDataSrc; safeCallEE(testDataSrc=new tFXData(FXDB, "EURUSD", "H1", false));
			tTS* testTS; safeCallEE(testTS=new tTS(testDataSrc, testTShistoryLen, testTSdate0, testTS_DT, myNN->scaleMin[0], myNN->scaleMax[0]));
			tDataSet* testSet; safeCallEE(testSet=new tDataSet(testTS, forecasterSampleLen, forecasterPredictionLen, forecasterFeaturesCnt, forecasterFeature, testBatchSamplesCnt));
			if (testWpid!=0&&testWtid!=0) {
				//-- if we specifed both testWpid and testWtid, then load training Weights('-1' stands for 'latest epoch')
				safeCallEE(persistor->LoadW(testWpid, testWtid, -1, myNN->weightsCntTotal, myNN->W));
			} else {
				//-- otherwise, testW pid/tid are those from current training session; also, use existing W from training
				testWpid=myNN->pid; testWtid=myNN->tid;
			}
			//-- Inference from Test Set (run + persist)
			safeCallEE(myNN->run(testSet));
			safeCallEE(persistor->SaveRun(myNN->pid, myNN->tid, 1, testWpid, testWtid, testSet->samplesCnt, forecasterFeaturesCnt, forecasterFeature, testSet->prediction0, testSet->target0));
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
		

		//-- destroy all objects (therefore all tDebugger* objects, therefore all empty debug files)
		delete FXDB;
		delete persistor;
		delete myNN;
		delete parms;

		delete dbg;
*/	}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed with exception: %s\n", 1, e.what());
		return -1;
	}

	system("pause");
	return 0;

}
