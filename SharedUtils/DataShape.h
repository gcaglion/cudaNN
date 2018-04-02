#pragma once
#include "Debugger.h"
#include "ParamMgr.h"

typedef struct sDataShape {

	tDebugger* dbg;

	int sampleLen;
	int predictionLen;
	int featuresCnt;

	EXPORT sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT sDataShape(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_=nullptr);

} tDataShape;
