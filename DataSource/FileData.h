// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"
#include "DataSource.h"
#include "../FileInfo/FileInfo.h"
#include "FileData_enums.h"

typedef struct sFileData : public sDataSource {
	tFileInfo* srcFile;
	int fieldSep;
	int featuresCnt;

	EXPORT sFileData(char* objName_, s0* objParent_, tFileInfo* srcFile_, int fieldSep_=COMMA_SEPARATOR, Bool calcBW_=false, int BWfeatureH_=1, int BWfeatureL_=2, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sFileData(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sFileData();

} tFileData;
