#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataFile.h"
#include "../MyEngines.h"
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
	int  pListLen;						//-- number of elements (of any type) found in array parameter
	char** pArrDesc;

	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDbg* dbg_=nullptr); //-- directly from command-line
	EXPORT ~sParamMgr();

	//-- enums
	EXPORT void getEnumVal(char* edesc, char* eVal, int* oVal);
	
	//-- generic
	template <typename T> EXPORT void get(T* opVal, const char* parmDesc, bool isenum=false) {
		strcpy_s(pDesc, parmDesc); 
		Trim(pDesc); 
		UpperCase(pDesc);

		get_(opVal, isenum);

/*		const char* pType=typeid((*opVal)).name();
		if (strcmp(pType, "int")==0) {
			get_(opVal, isenum);
		} else if ((strcmp(pType, "float")==0)||(strcmp(pType, "double")==0)) {
			get_(opVal);
		} else {
			get_(opVal);
		}
*/
	}
	//-- single value (int, double, char*, enum)
	EXPORT void get_(int* oparamVal, bool isenum=false);
	EXPORT void get_(numtype* oparamVal, bool isenum=false);
	EXPORT void get_(char* oparamVal, bool isenum=false);
	EXPORT void get_(int** oparamVal, bool isenum=false);
	EXPORT void get_(numtype** oparamVal, bool isenum=false);
	EXPORT void get_(char** oparamVal, bool isenum=false);
	EXPORT void ReadParamFromFile(int* oParamValue);
	EXPORT void ReadParamFromFile(numtype* oParamValue);
	EXPORT void ReadParamFromFile(char* oParamValue);

} tParamMgr;
