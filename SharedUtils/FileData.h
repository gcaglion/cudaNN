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

	EXPORT sFileData(tFileInfo* srcFile_, int fieldSep_=FILEDATA_SEP_COMMA, bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, tDbg* dbg_=nullptr);
	EXPORT sFileData(tParamMgr* parms, tDbg* dbg_=nullptr);

} tFileData;
