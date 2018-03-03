#pragma once

//-- Source Types
#define SOURCE_DATA_FROM_FXDB 0
#define SOURCE_DATA_FROM_FILE 1
#define SOURCE_DATA_FROM_MT4  2

typedef struct sDataSource {
	int type;
	int featuresCnt;
	bool calcBW;
	int BWfeatureH;
	int BWfeatureL;

#ifdef __cplusplus
	sDataSource(int type_, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2);
	sDataSource();
#endif

} tDataSource;
