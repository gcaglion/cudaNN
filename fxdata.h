#pragma once
#include "../CommonEnv.h"
#include "DBConnection.h"

// Bar data
#define MAXBARDATATYPES 6
#define OPEN   0
#define HIGH   1
#define LOW    2
#define CLOSE  3
#define VOLUME 4
#define OTHER  5

#define FX_SYMBOL_MAX_LEN 12
#define FX_TIMEFRAME_MAX_LEN 4

// Database retrieval properties
typedef struct sFXData {
	tDBConnection* FXDB;
	char Symbol[FX_SYMBOL_MAX_LEN];
	char TimeFrame[FX_TIMEFRAME_MAX_LEN];
	int IsFilled;
	int BarDataTypeCount;
	//int BarDataType[MAXBARDATATYPES];
	int* BarDataType;
#ifdef __cplusplus
	sFXData() {
		FXDB = new tDBConnection();
		BarDataType = (int*)malloc(MAXBARDATATYPES*sizeof(int));
	}
	sFXData(char* FXDBusername, char* FXDBpassword, char* FXDBconnstring, char* symbol_, char* tf_, int isFilled_) {
		FXDB = new tDBConnection(FXDBusername, FXDBpassword, FXDBconnstring);
		//BarDataType = (int*)malloc(MAXBARDATATYPES*sizeof(int));
		BarDataTypeCount=5; BarDataType= new int [MAXBARDATATYPES] { OPEN, HIGH, LOW, CLOSE, VOLUME };
		strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
		strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
		IsFilled=isFilled_;
	}

	~sFXData() {
		delete(FXDB);
		free(BarDataType);
	}
#endif
} tFXData;

// Bar structure
typedef struct {
	char NewDateTime[12 + 1];
	numtype Open;
	numtype High;
	numtype Low;
	numtype Close;
	numtype OpenD;
	numtype HighD;
	numtype LowD;
	numtype CloseD;
	numtype Volume;
	numtype VolumeD;
} tBar;
