#pragma once

//-- Source Types
#define SOURCE_DATA_FROM_FXDB 0
#define SOURCE_DATA_FROM_FILE 1
#define SOURCE_DATA_FROM_MT4  2

typedef struct sDataSource {
	int type;
	int featuresCnt;
	bool calcBW;
	int BWfeature[2];
} tDataSource;