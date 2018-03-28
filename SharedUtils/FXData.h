#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "DataSource.h"
#include "DBConnection.h"
#include "FXData_enums.h"

typedef struct sFXData : public sDataSource {
	tDBConnection* db;
	char Symbol[XML_MAX_PARAM_VAL_LEN];
	char TimeFrame[XML_MAX_PARAM_VAL_LEN];
	bool IsFilled;
	int BarDataType[FXDATA_FEATURESCNT];
#ifdef __cplusplus
	EXPORT sFXData(tDBConnection* db_, char* symbol_, char* tf_, bool isFilled_, tDebugger* dbg_=nullptr);
	EXPORT sFXData(tParmsSource* parms, tDebugger* dbg_=nullptr);
	EXPORT void sFXData_common(tDebugger* dbg_=nullptr);
#endif
} tFXData;

