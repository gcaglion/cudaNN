#pragma once
#include "Debugger.h"
#include "ParamMgr.h"

typedef struct sDataShape {

	tDebugger* dbg;

	int sampleLen;
	int predictionLen;
	int featuresCnt;

	EXPORT sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);

} tDataShape;
