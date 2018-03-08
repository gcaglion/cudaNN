#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "Generic.h"

//-- enumerators for ALL classes
#include "../TimeSerie/TimeSerie_enums.h"
//...

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
	int    parmDesc_depth;
	char*  parmDesc_Full;
	char** parmDesc_Step;

#ifdef __cplusplus
	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDbg* dbg_=nullptr); //-- directly from command-line
	EXPORT ~sParamMgr();

	EXPORT void sectionSet(const char* sectionLabel);
	EXPORT void sectionSetChild(const char* child);
	EXPORT void sectionSetParent();

	template <typename T> EXPORT void getx(T* opVal, const char* parmDesc, bool isenum=false, int* oListLen=nullptr) {
		char p[XML_MAX_SECTION_DESC_LEN];
		char ps[XML_MAX_SECTION_DESC_LEN+2];
		char pdesc[XML_MAX_PARAM_NAME_LEN];
		char pval[XML_MAX_PARAM_VAL_LEN];
		char parmPath_Full_Bkp[XML_MAX_PATH_LEN];

		//-- REWRITE COMMAND-LINE CHECK!

		//-- Split parmDesc
		parmDesc_depth=cslToArray(parmDesc, '.', parmDesc_Step);
		//-- backup parmPath_Full
		strcpy_s(parmPath_Full_Bkp, XML_MAX_PATH_LEN, parmPath_Full);
		//-- add all but last steps of parmDesc to parmPath
		for (int d=0; d<(parmDesc_depth-1); d++) {
			strcat(parmPath_Full, "."); strcat(parmPath_Full, parmDesc_Step[d]);
		}
		//-- parmSection is case-sensitive. parmDesc is not
		strcpy_s(pDesc, parmDesc_Step[parmDesc_depth-1]); Trim(pDesc);	UpperCase(pDesc);


		//-- 0. Split parmSection
		parmPath_depth=cslToArray(parmPath_Full, '.', parmPath_Step);
		//-- 1. Navigate to the first parameter of the Section
		rewind(ParamFile->handle);
		for (int d=0; d<parmPath_depth; d++) {
			sprintf_s(ps, XML_MAX_SECTION_DESC_LEN+2, "<%s>", parmPath_Step[d]);
			while (fscanf(ParamFile->handle, "%s", p)!=EOF) {
				if (strcmp(p, ps)==0) break;
			}
		}

		//-- 1. sequentially read all parameters until the end of the Section; return when found
		while (fscanf(ParamFile->handle, "%s = %[^\n]", pdesc, pval)!=EOF) {
			Trim(pdesc); UpperCase(pdesc);
			if (strcmp(pdesc, pDesc)==0) {
				//--- here we have the sought parameter value in pval
				getxx_(pval, opVal, isenum, oListLen);
				//-- restore before exiting
				strcpy_s(parmPath_Full, XML_MAX_PATH_LEN, parmPath_Full_Bkp);
				return;
			}
		}
		throwE("could not find parameter %s in section [%s]", 2, parmDesc, parmPath_Full);
	}
	template <typename T> EXPORT void getx(T* opVal, const char* sectionDesc, const char* parmDesc, bool isenum=false, int* oListLen=nullptr) {
		sectionSet(sectionDesc);
		getx(opVal, parmDesc, isenum, oListLen);
	}
	
	//-- specific, single value: int(with or without enums), numtype, char*
	EXPORT void getxx_(char* pvalS, int* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, bool* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, numtype* oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, char* oparamVal, bool isenum=false, int* oListLen=nullptr);
	//-- specific, arrays: int(with or without enums), numtype, char*
	EXPORT void getxx_(char* pvalS, int** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, bool** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, numtype** oparamVal, bool isenum=false, int* oListLen=nullptr);
	EXPORT void getxx_(char* pvalS, char** oparamVal, bool isenum=false, int* oListLen=nullptr);

	//-- enum(s) decoder
	void enumDecode(char* pName, char* pVal, int* opvalIdx);
#endif

} tParamMgr;
