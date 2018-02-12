#pragma once
#include "../CommonEnv.h"
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

// Database retrieval properties
typedef struct sFXData {
	tDBConnection* db;
	char Symbol[FX_SYMBOL_MAX_LEN];
	char TimeFrame[FX_TIMEFRAME_MAX_LEN];
	int IsFilled;
	int BarDataType[FXDATA_FEATURESCNT];
#ifdef __cplusplus
	sFXData(tDBConnection* db_, char* symbol_, char* tf_, int isFilled_) {
		db=db_;
		strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
		strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
		IsFilled=isFilled_;
	}

	~sFXData() {
	}
#endif
} tFXData;

