#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"

typedef struct sData {
	tDebugger* dbg;

	//-- shape
	int sampleLen;
	int predictionLen;
	int featuresCnt;

	//-- actions
	bool doTrain;
	bool doValidation;
	bool doTest;

	//-- TimeSeries and DataSets
	tTimeSerie* trainTS; tDataSet* trainDS;
	tTimeSerie* testTS; tDataSet* testDS;
	tTimeSerie* validTS; tDataSet* validDS;


	EXPORT sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sData();

private:
	void mallocSets();

} tData;