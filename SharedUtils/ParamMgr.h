#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataFile.h"
#include "../MyEngines.h"
#include "../Logger/Logger.h"
#include "../TimeSerie/TimeSerie.h"

#define MAXPARAMDESCLEN 20	// Max length for comma-separated lists of descriptions
#define ARRAY_PARAMETER_MAX_LEN	100
#define enumlist true

typedef struct sParamMgr {

	tDbg* dbg;

	//-- ini file and command-line overrides
	tFileInfo* ParamFile;
	char IniFileName[MAX_PATH];
	int  CLparamCount;
	char CLparamName[1000][100];
	char CLparamVal[1000][100];
	//--
	//char vParamName[MAXPARAMDESCLEN];

	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDbg* dbg_=nullptr); //-- directly from command-line

	//-- enums
	EXPORT void getEnumVal(char* edesc, char* eVal, int* oVal);
	
	//-- single value (int, double, char*, enum)
	EXPORT void getParam(char* paramName, double* oparamVal);
	EXPORT void getParam(char* paramName, char* oparamVal);
	EXPORT void getParam(char* paramName, int* oparamVal, bool isenum);
	EXPORT void ReadParamFromFile(char* pFileName, char* pParamName, int* oParamValue);
	EXPORT void ReadParamFromFile(char* pFileName, char* pParamName, double* oParamValue);
	EXPORT void ReadParamFromFile(char* pFileName, char* pParamName, char* oParamValue);

} tParamMgr;

