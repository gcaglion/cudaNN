#pragma once
#include "../CommonEnv.h"
#include "../BaseObj/BaseObj.h"
#include "DataSource_enums.h"
#include "../ParamMgr/ParamMgr.h"

typedef struct sDataSource : public sBaseObj {
	
	int type;

	int featuresCnt;
	bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(char* objName_, sBaseObj* objParent_, int type_, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, sDebuggerParms* dbgparms_=nullptr);
	sDataSource(char* objName_, sBaseObj* objParent_, tParmsSource* parms, sDebuggerParms* dbgparms_=nullptr);
	sDataSource(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_=nullptr);
	~sDataSource();
#endif

} tDataSource;
