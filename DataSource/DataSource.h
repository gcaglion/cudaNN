#pragma once
#include "../CommonEnv.h"
#include "DataSource_enums.h"
#include "../ParamMgr/ParamMgr.h"

typedef struct sDataSource : public sBaseObj {
	
	int type;

	int featuresCnt;
	bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(int type_, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, tDebugger* dbg_=nullptr);
	sDataSource(tParmsSource* parms, tDebugger* dbg_=nullptr);
	sDataSource(tDebugger* dbg_=nullptr);
	~sDataSource();
#endif

} tDataSource;
