#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataFile.h"
//#include "../MyEngines.h"
#include "../Logger/Logger.h"
#include "../TimeSerie/TimeSerie.h"
#include <typeinfo>

#define MAX_PARAMS_CNT 100
#define MAX_PARAMDESC_LEN 100	
#define ARRAY_PARAMETER_MAX_ELEMS	64	// Max length for comma-separated lists of descriptions
#define enumlist true

typedef struct sParamMgr {

	tDbg* dbg;

	//-- ini file and command-line overrides
	tFileInfo* ParamFile;
	int  CLparamCount;
	char CLparamName[MAX_PARAMS_CNT][MAX_PARAMDESC_LEN];
	char CLparamVal[MAX_PARAMS_CNT][MAX_PARAMDESC_LEN];
	//--
	char pDesc[MAX_PARAMDESC_LEN];
	char pListDesc[MAX_PARAMDESC_LEN];	//-- string containing the list of values (of any type) in an array parameter
	char** pArrDesc;

	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDbg* dbg_=nullptr); //-- directly from command-line
	EXPORT ~sParamMgr();

	//-- enums
	EXPORT void getEnumVal(char* edesc, char* eVal, int* oVal);
	
	//-- generic
	template <typename T> EXPORT void get(T* opVal, const char* parmDesc, bool isenum=false, int* oListLen=nullptr) {
		strcpy_s(pDesc, parmDesc); 
		Trim(pDesc); 
		UpperCase(pDesc);

		get_(opVal, isenum, oListLen);

	}
	//-- single value: int(with or without enums), numtype, char*
	EXPORT void get_(int* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void get_(numtype* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void get_(char* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void get_(bool* oparamVal, bool isenum=false, int* oListLen=nullptr);
	//-- arrays: int(with or without enums), numtype, char*
	EXPORT void get_(int** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void get_(numtype** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void get_(char** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void ReadParamFromFile(int* oParamValue);
	EXPORT void ReadParamFromFile(numtype* oParamValue);
	EXPORT void ReadParamFromFile(char* oParamValue);
	EXPORT void ReadParamFromFile(bool* oParamValue);

} tParamMgr;
