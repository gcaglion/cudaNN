#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "ParamMgr.h"

typedef struct sDataSource {
	
	tDebugger* dbg;

	int type;

	int featuresCnt;
	bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(int type_, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, tDebugger* dbg_=nullptr);
	sDataSource(tParmsSource* parms, tDebugger* dbg_=nullptr);
	sDataSource(){}
#endif

} tDataSource;
