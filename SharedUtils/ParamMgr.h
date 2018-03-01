#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataFile.h"
#include "../MyEngines.h"
#include "../Logger/Logger.h"
#include "../TimeSerie/TimeSerie.h"
#include <typeinfo>

#define MAX_CL_PARAMS_CNT 100
#define MAXPARAMDESCLEN 100	
#define ARRAY_PARAMETER_MAX_LEN	100	// Max length for comma-separated lists of descriptions
#define enumlist true

typedef struct sParamMgr {

	tDbg* dbg;

	//-- ini file and command-line overrides
	tFileInfo* ParamFile;
	int  CLparamCount;
	char CLparamName[1000][100];
	char CLparamVal[1000][100];
	//--
	char pDesc[MAXPARAMDESCLEN];

	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDbg* dbg_=nullptr); //-- directly from command-line

	//-- enums
	EXPORT void getEnumVal(char* edesc, char* eVal, int* oVal);
	
	//-- generic
	template <typename T> EXPORT void get(T* opVal, const char* parmDesc, bool isenum=false) {
		strcpy_s(pDesc, parmDesc); Trim(pDesc); UpperCase(pDesc);
		if (strcmp(typeid((*opVal)).name(), "int")==0) {
			get_(opVal, isenum);
		} else if ((strcmp(typeid((*opVal)).name(), "float")==0)||(strcmp(typeid((*opVal)).name(), "double")==0)) {
			get_(opVal);
		} else {
			get_(opVal);
		}

	}
	//-- single value (int, double, char*, enum)
	EXPORT void get_(numtype* oparamVal, bool isenum=false);
	EXPORT void get_(char* oparamVal, bool isenum=false);
	EXPORT void get_(int* oparamVal, bool isenum=false);
	EXPORT void ReadParamFromFile(int* oParamValue);
	EXPORT void ReadParamFromFile(numtype* oParamValue);
	EXPORT void ReadParamFromFile(char* oParamValue);

} tParamMgr;
