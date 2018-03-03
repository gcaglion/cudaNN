// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"
#include "DataSource.h"
#include "FileInfo.h"

//-- field separators
#define FILEDATA_SEP_COMMA 0
#define FILEDATA_SEP_TAB 1
#define FILEDATA_SEP_SPACE 2

typedef struct sFileData : public sDataSource {
	tFileInfo* srcFile;
	int fieldSep;
	int featuresCnt;

	sFileData(tFileInfo* srcFile_, int fieldSep_=FILEDATA_SEP_COMMA, bool calcBW_=false, int* BWfeature=nullptr);

} tFileData;
