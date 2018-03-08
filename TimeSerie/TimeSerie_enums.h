#pragma once
#include "../CommonEnv.h"

//-- Data Tranformations
#define DT_NONE		 0
#define DT_DELTA	 1
#define DT_LOG		 2
#define DT_DELTALOG	 3

//-- Statistical Features
#define TSF_MEAN 0
#define TSF_MAD 1
#define TSF_VARIANCE 2
#define TSF_SKEWNESS 3
#define TSF_KURTOSIS 4
#define TSF_TURNINGPOINTS 5
#define TSF_SHE 6
#define TSF_HISTVOL 7

//-- data sets
#define TRAIN_SET 0
#define TEST_SET  1
#define VALID_SET 2

EXPORT int decode(char* paramName, char* stringToCheck) {
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
	
	return -1;
}
