#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataSource.h"
#include "DBConnection.h"

// Bar data
#define FXDATA_FEATURESCNT 5	//-- OHLCV
#define OPEN 0
#define HIGH 1
#define LOW 2
#define CLOSE 3
#define VOLUME 4

#define FX_SYMBOL_MAX_LEN 12
#define FX_TIMEFRAME_MAX_LEN 4

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

