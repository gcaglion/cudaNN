#pragma once
#include "../CommonEnv.h"
#include "DataSource.h"
#include "../DBConnection/DBConnection.h"
#include "../ParamMgr/ParamMgr.h"
#include "FXData_enums.h"

#define FX_SYMBOL_MAXLEN 8
#define FX_TIMEFRAME_MAXLEN 4

typedef struct sFXData : public sDataSource {
	tDBConnection* db=nullptr;
	char Symbol[FX_SYMBOL_MAXLEN];
	char TimeFrame[FX_TIMEFRAME_MAXLEN];
	Bool IsFilled;
#ifdef __cplusplus
	EXPORT sFXData(char* objName_, s0* objParent_, tDBConnection* db_, char* symbol_, char* tf_, Bool isFilled_, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sFXData(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sFXData();
	EXPORT void sFXData_common(sDebuggerParms* dbgparms_=nullptr);
#endif
} tFXData;

