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

// Database retrieval properties
typedef struct sFXData {
	tDBConnection* FXDB;
	char Symbol[12];
	char TimeFrame[4];
	int IsFilled;
	int BarDataTypeCount;
	//int BarDataType[MAXBARDATATYPES];
	int* BarDataType;
#ifdef __cplusplus
	sFXData() {
		FXDB = new tDBConnection();
		BarDataType = (int*)malloc(MAXBARDATATYPES*sizeof(int));
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
