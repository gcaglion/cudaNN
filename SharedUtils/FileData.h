// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"
#include "DataSource.h"
#include "FileInfo.h"
#include "FileData_enums.h"

typedef struct sFileData : public sDataSource {
	tFileInfo* srcFile;
	int fieldSep;
	int featuresCnt;

	EXPORT sFileData(tFileInfo* srcFile_, int fieldSep_=COMMA_SEPARATOR, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, tDebugger* dbg_=nullptr);
	EXPORT sFileData(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);

} tFileData;
