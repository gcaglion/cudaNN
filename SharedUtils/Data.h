#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "ParamMgr.h"
#include "../TimeSerie/TimeSerie.h"

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
	tTimeSerie** ts;
	tDataSet**   ds;

	EXPORT sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sData();

private:
	void mallocSets();

} tData;