#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "ParamMgr.h"

//-- Source Types
#define SOURCE_DATA_FROM_FXDB 0
#define SOURCE_DATA_FROM_FILE 1
#define SOURCE_DATA_FROM_MT4  2

typedef struct sDataSource {
	
	tDbg* dbg;

	int type;

	int featuresCnt;
	bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(int type_, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, tDbg* dbg_=nullptr);
	sDataSource(tParamMgr* parms, tDbg* dbg_=nullptr);
	sDataSource(){}
#endif

} tDataSource;
