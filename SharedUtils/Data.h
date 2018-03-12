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

	//-- TimeSeries and DataSets
	tTimeSerie** ts;
	tDataSet**   ds;

	EXPORT sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sData(tParmsSource* parms, tDebugger* dbg_=nullptr);
	~sData();
private:
	void mallocSets();

} tData;