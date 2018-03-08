#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataSource.h"
#include "DBConnection.h"
#include "FXData_enums.h"

typedef struct sFXData 
#ifdef __cplusplus
	: public sDataSource
#endif
{
	tDBConnection* db;
	char Symbol[FX_SYMBOL_MAX_LEN];
	char TimeFrame[FX_TIMEFRAME_MAX_LEN];
	int IsFilled;
	int BarDataType[FXDATA_FEATURESCNT];
#ifdef __cplusplus
	EXPORT sFXData(tDBConnection* db_, char* symbol_, char* tf_, int isFilled_, tDbg* dbg_=nullptr);
	EXPORT sFXData(tParamMgr* parms, tDbg* dbg_=nullptr);
	EXPORT void sFXData_common(tDbg* dbg_=nullptr);
#endif
} tFXData;

