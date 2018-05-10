#pragma once
#include "../CommonEnv.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"
#include "../ParamMgr/ParamMgr.h"

//-- Actions on data
#define TRAIN 0
#define TEST  1
#define VALID 2

typedef struct sDataShape : public s0 {

	//-- basic properties
	int sampleLen;
	int predictionLen;
	int featuresCnt;

	//-- these are set within Cores
	int inputCnt;
	int outputCnt;

	EXPORT sDataShape(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sDataShape(char* objName_, s0* objParent_, int sampleLen_, int predictionLen_, int featuresCnt_, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sDataShape();

} tDataShape;

typedef struct sData : public s0 {

	//-- shape
	tDataShape* shape=nullptr;

	//-- actions
	Bool ActionDo[3];
	char ActionDesc[3][XML_MAX_SECTION_DESC_LEN]={"Train","Test","Validation"};

	//-- DataSets (each include its own source TimeSerie)
	tDataSet* ds[3];

	EXPORT sData(char* objName_, s0* objParent_, tDataShape* shape_=nullptr, Bool doTrain=true, Bool doTest=true, Bool doValidation=false, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sData(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sData();

} tData;