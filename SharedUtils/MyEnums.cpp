#include "MyEnums.h"
/*
EXPORT bool decode(char* stringToCheck, tParm* enumParm, int* oValsCnt, int* oVal) {
	bool ret=false;

	if (strcmp(enumName, "DATASOURCE.TYPE")==0) {
		if (strcmp(stringToCheck, "FXDB_SOURCE")==0) { (*oVal)=FXDB_SOURCE; ret=true; }
		if (strcmp(stringToCheck, "FILE_SOURCE")==0) { (*oVal)=FILE_SOURCE; ret=true; }
		if (strcmp(stringToCheck, "MT4_SOURCE")==0) { (*oVal)=MT4_SOURCE; ret=true; }
	} else if (strcmp(enumName, "LEVEL")==0) {
		if (strcmp(stringToCheck, "DBG_LEVEL_ERR")==0) { (*oVal)=DBG_LEVEL_ERR; ret=true; }
		if (strcmp(stringToCheck, "DBG_LEVEL_STD")==0) { (*oVal)=DBG_LEVEL_STD; ret=true; }
		if (strcmp(stringToCheck, "DBG_LEVEL_DET")==0) { (*oVal)=DBG_LEVEL_DET; ret=true; }
	} else if (strcmp(enumName, "DESTINATION")==0) {
		if (strcmp(stringToCheck, "DBG_DEST_SCREEN")==0) { (*oVal)=DBG_DEST_SCREEN; ret=true; }
		if (strcmp(stringToCheck, "DBG_DEST_FILE")==0) { (*oVal)=DBG_DEST_FILE; ret=true; }
		if (strcmp(stringToCheck, "DBG_DEST_BOTH")==0) { (*oVal)=DBG_DEST_BOTH; ret=true; }
	} else if (strcmp(enumName, "FILEDATA.SEPARATOR")==0) {
		if (strcmp(stringToCheck, "FILEDATA_SEP_COMMA")==0) { (*oVal)=FILEDATA_SEP_COMMA; ret=true; }
		if (strcmp(stringToCheck, "FILEDATA_SEP_SPACE")==0) { (*oVal)=FILEDATA_SEP_SPACE; ret=true; }
		if (strcmp(stringToCheck, "FILEDATA_SEP_TAB")==0) { (*oVal)=FILEDATA_SEP_TAB; ret=true; }
	} else if (strcmp(enumName, "FILE.OPEN.MODE")==0) {
		if (strcmp(stringToCheck, "FILE_MODE_READ")==0) { (*oVal)=FILE_MODE_READ; ret=true; }
		if (strcmp(stringToCheck, "FILE_MODE_WRITE")==0) { (*oVal)=FILE_MODE_WRITE; ret=true; }
		if (strcmp(stringToCheck, "FILE_MODE_APPEND")==0) { (*oVal)=FILE_MODE_APPEND; ret=true; }
	} else if (strcmp(enumName, "FXDATA.FEATURES")==0) {
		if (strcmp(stringToCheck, "FXDATA_OPEN")==0) { (*oVal)=FXDATA_OPEN; ret=true; }
		if (strcmp(stringToCheck, "FXDATA_LOW")==0) { (*oVal)=FXDATA_LOW; ret=true; }
		if (strcmp(stringToCheck, "FXDATA_HIGH")==0) { (*oVal)=FXDATA_HIGH; ret=true; }
		if (strcmp(stringToCheck, "FXDATA_CLOSE")==0) { (*oVal)=FXDATA_CLOSE; ret=true; }
		if (strcmp(stringToCheck, "FXDATA_VOLUME")==0) { (*oVal)=FXDATA_VOLUME; ret=true; }
	} else if (strcmp(enumName, "DATA.TRANSFORMATION")==0) {
		if (strcmp(stringToCheck, "DT_NONE")==0) { (*oVal)=DT_NONE; ret=true; }
		if (strcmp(stringToCheck, "DT_DELTA")==0) { (*oVal)=DT_DELTA; ret=true; }
		if (strcmp(stringToCheck, "DT_LOG")==0) { (*oVal)=DT_LOG; ret=true; }
		if (strcmp(stringToCheck, "DT_DELTALOG")==0) { (*oVal)=DT_DELTALOG; ret=true; }
	} else if (strcmp(enumName, "STATISTICAL.FEATURES")==0) {
		if (strcmp(stringToCheck, "TSF_MEAN")==0) { (*oVal)=TSF_MEAN; ret=true; }
		if (strcmp(stringToCheck, "TSF_MAD")==0) { (*oVal)=TSF_MAD; ret=true; }
		if (strcmp(stringToCheck, "TSF_VARIANCE")==0) { (*oVal)=TSF_VARIANCE; ret=true; }
		if (strcmp(stringToCheck, "TSF_SKEWNESS")==0) { (*oVal)=TSF_SKEWNESS; ret=true; }
		if (strcmp(stringToCheck, "TSF_KURTOSIS")==0) { (*oVal)=TSF_KURTOSIS; ret=true; }
		if (strcmp(stringToCheck, "TSF_TURNINGPOINTS")==0) { (*oVal)=TSF_TURNINGPOINTS; ret=true; }
		if (strcmp(stringToCheck, "TSF_SHE")==0) { (*oVal)=TSF_SHE; ret=true; }
		if (strcmp(stringToCheck, "TSF_HISTVOL")==0) { (*oVal)=TSF_HISTVOL; ret=true; }
	} else if (strcmp(enumName, "DATA.SET")==0) {
		if (strcmp(stringToCheck, "TRAIN_SET")==0) { (*oVal)=TRAIN_SET; ret=true; }
		if (strcmp(stringToCheck, "TEST_SET")==0) { (*oVal)=TEST_SET; ret=true; }
		if (strcmp(stringToCheck, "VALID_SET")==0) { (*oVal)=VALID_SET; ret=true; }
	}
	return ret;
}
*/