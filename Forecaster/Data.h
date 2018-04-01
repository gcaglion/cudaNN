#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "../SharedUtils/DataShape.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"

//-- Actions on data
#define TRAIN 0
#define TEST  1
#define VALID 2

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