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
	bool IsFilled;
#ifdef __cplusplus
	EXPORT sFXData(tDBConnection* db_, char* symbol_, char* tf_, bool isFilled_, tDebugger* dbg_=nullptr);
	EXPORT sFXData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sFXData();
	EXPORT void sFXData_common(tDebugger* dbg_=nullptr);
#endif
} tFXData;

