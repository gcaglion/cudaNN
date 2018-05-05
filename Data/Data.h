#pragma once
#include "../CommonEnv.h"
#include "../TimeSerie/TimeSerie.h"
#include "../TimeSerie/DataSet.h"
#include "../ParamMgr/ParamMgr.h"

//-- Actions on data
#define TRAIN 0
#define TEST  1
#define VALID 2

typedef struct sDataShape : public sBaseObj {

	//-- basic properties
	int sampleLen;
	int predictionLen;
	int featuresCnt;

	//-- these are set within Cores
	int inputCnt;
	int outputCnt;

	EXPORT sDataShape(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sDataShape(char* objName_, sBaseObj* objParent_, int sampleLen_, int predictionLen_, int featuresCnt_, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sDataShape();

} tDataShape;

typedef struct sData : public sBaseObj {

	//-- shape
	tDataShape* shape=nullptr;

	//-- actions
	bool ActionDo[3];
	char ActionDesc[3][XML_MAX_SECTION_DESC_LEN]={"Train","Test","Validation"};

	//-- DataSets (each include its own source TimeSerie)
	tDataSet* ds[3];

	EXPORT sData(char* objName_, sBaseObj* objParent_, tDataShape* shape_=nullptr, bool doTrain=true, bool doTest=true, bool doValidation=false, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sData(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sData();

} tData;