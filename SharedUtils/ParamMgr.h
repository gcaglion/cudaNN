#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "FileData.h"
#include "../TimeSerie/TimeSerie.h"
#include <typeinfo>
#include "../Logger/Logger.h"

#define MAX_PARAMS_CNT 100
#define MAX_PARAMDESC_LEN 100	
#define ARRAY_PARAMETER_MAX_ELEMS	64	// Max length for comma-separated list of descriptions
#define enumlist true

//==== XML stuff
#define XML_MAX_PATH_DEPTH 128
#define XML_MAX_SECTION_DESC_LEN 64
#define XML_MAX_PATH_LEN XML_MAX_PATH_DEPTH*XML_MAX_SECTION_DESC_LEN
#define XML_MAX_PARAM_NAME_LEN	128
#define XML_MAX_PARAM_VAL_LEN	128

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

	//==== XML stuff
	int    parmPath_depth;
	char*  parmPath_Full;
	char** parmPath_Step;

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


	//-- generic
	template <typename T> EXPORT void getx(T* opVal, const char* parmDesc, bool isenum=false, int* oListLen=nullptr) {
		char p[XML_MAX_SECTION_DESC_LEN]; 
		char ps[XML_MAX_SECTION_DESC_LEN+2];
		char pdesc[XML_MAX_PARAM_NAME_LEN];
		char pval[XML_MAX_PARAM_VAL_LEN];
		
		const char* pType=typeid(T).name();

		//-- parmSection is case-sensitive. parmDesc is not
		strcpy_s(pDesc, parmDesc); Trim(pDesc);	UpperCase(pDesc);

		//-- REWRITE COMMAND-LINE CHECK!

		//-- 0. Split parmSection
		parmPath_depth=cslToArray(parmPath_Full, '.', parmPath_Step);
		//-- 1. Navigate to the first parameter of the Section
		rewind(ParamFile->handle);
		for (int d=0; d<parmPath_depth; d++) {
			sprintf_s(ps, XML_MAX_SECTION_DESC_LEN+2, "<%s>", parmPath_Step[d]);
			while(fscanf(ParamFile->handle, "%s", p)!=EOF) {
				if (strcmp(p, ps)==0) break;
			}
		}
		//-- 1. sequentially read all parameters until the end of the Section; return when found
		while (fscanf(ParamFile->handle, "%s = %s ", pdesc, pval)!=EOF) {
			Trim(pdesc); UpperCase(pdesc);
			if (strcmp(pdesc, pDesc)==0) {
				//...
			}
		}



	}
	//-- single value: int(with or without enums), numtype, char*
	EXPORT void getxx_(char* pvalS, int* oparamVal, bool isenum=false, int* oListLen=nullptr){
	}
	EXPORT void getxx_(bool* oparamVal, bool isenum=false, int* oListLen=nullptr){}
	EXPORT void getxx_(numtype* oparamVal, bool isenum=false, int* oListLen=nullptr){}
	EXPORT void getxx_(char* oparamVal, bool isenum=false, int* oListLen=nullptr){}


} tParamMgr;

/*#define getParm(p, varType, varLen, varName, varLabel) \
	const char* typeDesc=typeid(varType).name(); \
	printf("%s\n", typeDesc); \
		varType varName; \
	if(strcmp(typeDesc,"char")==0) { \
	} else if(strcmp((typeDesc),"int")==0){ \
		p->get(&varName, varLabel); \
	} else if(strcmp(typeDesc,"float")==0||strcmp(typeDesc,"double")==0) { \
	} else if(strcmp(typeDesc,"char*")==0) { \
	} else if(strcmp(typeDesc,"int*")==0){ \
	} else if(strcmp(typeDesc,"float*")==0||strcmp(typeDesc,"double*")==0) { \
	} else if(strcmp(typeDesc,"char**")==0) { \
	} else{ \
	}
*/
