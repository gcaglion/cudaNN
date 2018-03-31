#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"

//-- Actions on data
#define TRAIN 0
#define TEST  1
#define VALID 2

typedef struct sData {
	tDebugger* dbg;

	//-- shape
	int sampleLen;
	int predictionLen;
	int featuresCnt;

	//-- actions
	bool ActionDo[3];
	char ActionDesc[3][XML_MAX_SECTION_DESC_LEN]={"Train","Test","Validation"};

	//-- DataSets (each include its own source TimeSerie)
	tDataSet* ds[3];

	EXPORT sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sData();

} tData;