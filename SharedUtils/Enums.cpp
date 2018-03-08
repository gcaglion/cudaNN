#include "Enums.h"

EXPORT int decode(char* paramName, char* stringToCheck, int* oCode) {
	//-- Data Tranformations
	if (strcmp(paramName, "DATATRANSFORMATION")==0) {
		if (strcmp(stringToCheck, "DT_NONE")==0) return DT_NONE;
		if (strcmp(stringToCheck, "DT_DELTA")==0) return DT_DELTA;
		if (strcmp(stringToCheck, "DT_LOG")==0) return DT_LOG;
		if (strcmp(stringToCheck, "DT_DELTALOG")==0) return DT_DELTALOG;
	}
	//-- Statistical Features
	if (strcmp(paramName, "STATISTICALFEATURES")==0) {
		if (strcmp(stringToCheck, "TSF_MEAN")==0) return TSF_MEAN;
		if (strcmp(stringToCheck, "TSF_MAD")==0) return TSF_MAD;
		if (strcmp(stringToCheck, "TSF_VARIANCE")==0) return TSF_VARIANCE;
		if (strcmp(stringToCheck, "TSF_SKEWNESS")==0) return TSF_SKEWNESS;
		if (strcmp(stringToCheck, "TSF_KURTOSIS")==0) return TSF_KURTOSIS;
		if (strcmp(stringToCheck, "TSF_SHE")==0) return TSF_SHE;
		if (strcmp(stringToCheck, "TSF_HISTVOL")==0) return TSF_HISTVOL;
	}
	//-- data sets
	if (strcmp(paramName, "DATASET")==0) {
		if (strcmp(stringToCheck, "TRAIN_SET")==0) return TRAIN_SET;
		if (strcmp(stringToCheck, "TEST_SET")==0) return TEST_SET;
		if (strcmp(stringToCheck, "VALID_SET")==0) return VALID_SET;
	}
	//-- data source types
	if (strcmp(paramName, "DATASOURCETYPE")==0) {
		if (strcmp(stringToCheck, "SOURCE_DATA_FROM_FXDB")==0) return SOURCE_DATA_FROM_FXDB;
		if (strcmp(stringToCheck, "SOURCE_DATA_FROM_FILE")==0) return SOURCE_DATA_FROM_FILE;
		if (strcmp(stringToCheck, "SOURCE_DATA_FROM_MT4")==0) return SOURCE_DATA_FROM_MT4;
	}
	
	return -1;

#define MAX_ENUMS_CNT 32
#define kkk(pNameVar, pNameStr, pCheckStr, pValStrArray, pValVarArray){ \
	for(int i=0; i<MAX_ENUMS_CNT; i++) { \
#ifndef (pValArrayVar)[i])
	break;
#endif
		if(strcmp((pNameVar, (pValArrayVar)[i])==0)) return (pValVarArray)[i]; \
	} \
}
