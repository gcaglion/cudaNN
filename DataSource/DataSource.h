#pragma once
#include "../CommonEnv.h"
#include "../s0/s0.h"
#include "DataSource_enums.h"
#include "../ParamMgr/ParamMgr.h"

typedef struct sDataSource : public s0 {
	
	int type;

	int featuresCnt;
	Bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(char* objName_, s0* objParent_, int type_, Bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, sDebuggerParms* dbgparms_=nullptr);
	sDataSource(char* objName_, s0* objParent_, tParmsSource* parms, sDebuggerParms* dbgparms_=nullptr);
	sDataSource(char* objName_, s0* objParent_, sDebuggerParms* dbgparms_=nullptr);
	~sDataSource();
#endif

} tDataSource;
