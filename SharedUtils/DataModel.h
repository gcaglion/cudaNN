#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "ParamMgr.h"

typedef struct sDataModel {
	tDbg* dbg;

	//-- shape
	int sampleLen;
	int predictionLen;
	int featuresCnt;
	//-- actions
	bool doTrain;
	bool doTestOnTrain;
	bool doValid;
	bool doTest;

	EXPORT sDataModel(int sampleLen_, int predictionLen_, int featuresCnt_, tDbg* dbg_=nullptr);
	EXPORT sDataModel(tParamMgr* parms, tDbg* dbg_=nullptr);
	~sDataModel();

} tDataModel;