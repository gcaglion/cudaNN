#pragma once
#include "../CommonEnv.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"

//-- Actions on data
#define TRAIN 0
#define TEST  1
#define VALID 2

typedef struct sDataShape {

	tDebugger* dbg;

	int sampleLen;
	int predictionLen;
	int featuresCnt;

	EXPORT sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT sDataShape(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);

} tDataShape;

typedef struct sData {
	tDebugger* dbg;

	//-- shape
	tDataShape* shape;

	//-- actions
	bool ActionDo[3];
	char ActionDesc[3][XML_MAX_SECTION_DESC_LEN]={"Train","Test","Validation"};

	//-- DataSets (each include its own source TimeSerie)
	tDataSet* ds[3];

	EXPORT sData(tDataShape* shape_=nullptr, bool doTrain=true, bool doTest=true, bool doValidation=false, tDebugger* dbg_=nullptr);
	EXPORT sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sData();

} tData;