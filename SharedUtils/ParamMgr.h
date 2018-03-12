#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "Generic.h"
#include "Enums.h"

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

typedef struct sXMLelement {
	int depth;
	char** step;
	bool found;

	sXMLelement() {
		depth=0;
		step=(char**)malloc(XML_MAX_PATH_DEPTH*sizeof(char*)); for (int i=0; i<XML_MAX_PATH_DEPTH; i++) step[i]=(char*)malloc(XML_MAX_SECTION_DESC_LEN);
	}
	~sXMLelement() {
		for (int i=0; i<XML_MAX_PATH_DEPTH; i++) step[i];
		free(step);
	}

	void setFromDesc(const char* desc, bool fromRoot=false) {
		depth=cslToArray(desc, '.', step);
	}
	void copyTo(sXMLelement* destElement) {
		for (int d=0; d<depth; d++) {
			memcpy_s(destElement->step[d], XML_MAX_SECTION_DESC_LEN, step[d], XML_MAX_SECTION_DESC_LEN);
			destElement->depth=depth;
		}
		//memcpy_s((void*)destElement, sizeof(sXMLelement), this, sizeof(sXMLelement));
	}
	void appendTo(sXMLelement* destParm) {
		for (int d=0; d<depth; d++) {
			strcpy_s(destParm->step[destParm->depth+d], XML_MAX_SECTION_DESC_LEN, step[d]);
			destParm->depth++;
		}
	}

} tXMLelement;

typedef struct sKey : sXMLelement {
	char vLine[XML_MAX_SECTION_DESC_LEN+3];
	char keyStepTagStart[XML_MAX_SECTION_DESC_LEN+2];
	char keyStepTagEnd[XML_MAX_SECTION_DESC_LEN+3];

	bool find(tFileInfo* parmsFile) {

		//-- backup
		parmsFile->savePos();
		//--
		for (int d=0; d<depth; d++) {
			sprintf_s(keyStepTagStart, XML_MAX_SECTION_DESC_LEN+2, "<%s>", step[d]);
			sprintf_s(keyStepTagEnd, XML_MAX_SECTION_DESC_LEN+3, "</%s>", step[d]);

			//-- locate tag start
			found=false;
			while (fscanf_s(parmsFile->handle, "%s", vLine, XML_MAX_SECTION_DESC_LEN+3)!=EOF) {
				if (strcmp(vLine, keyStepTagStart)==0) {
					found=true;
					break;
				}
			}
		}
		//-- restore
		if (!found) parmsFile->restorePos();
		//--
		return (found);
	}

} tKey;
typedef struct sParm : sXMLelement {
	char name[XML_MAX_PARAM_NAME_LEN];
	char val[XML_MAX_PARAM_VAL_LEN];

	bool find(tFileInfo* parmsFile) {

		//-- backup
		parmsFile->savePos();
		//--
		//-- locate param name
		found=false;
		
		while (fscanf_s(parmsFile->handle, "%s = %[^\n]", name, XML_MAX_PARAM_NAME_LEN, val, XML_MAX_PARAM_VAL_LEN)!=EOF) {
			if (strcmp(name, step[depth-1])==0) {
				found=true;
				break;
			}
		}
		//-- restore
		if (!found) parmsFile->restorePos();
		//--
		return (found);
	}
} tParm;

typedef struct sParmsSource {

	tDebugger* dbg;

	tFileInfo* parmsFile;
	int CLoverridesCnt;
	char** CLoverride;

	tKey* currentKey;
	tKey* soughtKey;
	tKey* tmpKey;
	tParm* currentParm;
	tParm* soughtParm;
	tParm* tmpParm;

	EXPORT sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_=nullptr);
	void newDebugger(tDebugger* dbg_);

	EXPORT bool gotoKey(char* soughtKeyDesc, bool fromRoot=false, bool ignoreError=false) {

		soughtKey->setFromDesc(soughtKeyDesc, fromRoot);

		if (fromRoot) rewind(parmsFile->handle);

		//-- go to current key start
		if (soughtKey->find(parmsFile)) {
			//-- if found, set it as current
			if (fromRoot) {
				soughtKey->copyTo(currentKey);
			} else{
				soughtKey->appendTo(currentKey);
			}
		} else {
			throwE("could not find key", 0);
		}
		return true;
	}


	template <typename T> EXPORT void get(T* opVal, const char* soughtParmDesc, bool fromRoot=false, bool isenum=false, int* oListLen=nullptr) {

		soughtParm->setFromDesc(soughtParmDesc, fromRoot);

		if (fromRoot) rewind(parmsFile->handle);
		if(soughtParm->find(parmsFile)) {
			get_(  )
		}


		//-- set current key & filePos (restore from backup if find() failed, from current pos if successful)

		//-- if found key, lookup parm within key
	}

	//-- specific, single value: int(with or without enums), numtype, char*
	void get_(char* pvalS, int* oparamVal, bool isenum, int* oListLen) {
		if (isenum) {
			decode(soughtParm->val, pvalS, oparamVal);
		} else {
			(*oparamVal)=atoi(pvalS);

		}
	}

} tParmsSource;

//-- specific, single value: int(with or without enums), numtype, char*
void sParamMgr::getxx_(char* pvalS, int* oparamVal, bool isenum, int* oListLen) {
	if (isenum) {
		decode(pDesc, pvalS, oparamVal);
	} else {
		(*oparamVal)=atoi(pvalS);
	}
}
void sParamMgr::getxx_(char* pvalS, bool* oparamVal, bool isenum, int* oListLen) {
	Trim(pvalS); UpperCase(pvalS);
	(*oparamVal)=(strcmp(pvalS, "TRUE")==0);
}
void sParamMgr::getxx_(char* pvalS, numtype* oparamVal, bool isenum, int* oListLen) {
	(*oparamVal)=(numtype)atof(pvalS);
}
void sParamMgr::getxx_(char* pvalS, char* oparamVal, bool isenum, int* oListLen) {
	strcpy_s(oparamVal, XML_MAX_PARAM_VAL_LEN, pvalS);
}
//-- specific, arrays: int(with or without enums), numtype, char*
void sParamMgr::getxx_(char* pvalS, int** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], &(*oparamVal)[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, bool** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, numtype** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, char** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}

/*
typedef struct sParamMgr {

	tDebugger* dbg;

	//-- ini file and command-line overrides
	tFileInfo* ParamFile;
	int  CLparamCount;
	char CLparamName[MAX_PARAMS_CNT][MAX_PARAMDESC_LEN];
	char CLparamVal[MAX_PARAMS_CNT][MAX_PARAMDESC_LEN];
	//--
	char pDesc[MAX_PARAMDESC_LEN];
	char pListDesc[MAX_PARAMDESC_LEN];	//-- string containing the list of values (of any type) in an array parameter
	char** pArrDesc;

	//-- here we keep current, old and sought Key in open file
	//tXmlKey* currKey;
	//tXmlKey* oldKey;
	tXmlKey* soughtKey;

#ifdef __cplusplus
	//-- constructors / destructors
	EXPORT sParamMgr(tFileInfo* ParamFile_=nullptr, int argc=0, char* argv[]=nullptr, tDebugger* dbg_=nullptr); //-- directly from command-line
	EXPORT ~sParamMgr();

	//--- create a new Debugger from ParmFile
	void sParamMgr::newDebugger(tDebugger* dbg_);


	EXPORT bool sectionSet(const char* keyDesc, bool fromRoot=false, bool ignoreError=false);

	template <typename T> EXPORT void getx(T* opVal, const char* parmDesc, bool fromRoot=false, bool isenum=false, int* oListLen=nullptr) {
		soughtKey= new tXmlKey();

	}

	template <typename T> EXPORT void getx(T* opVal, const char* parmDesc, bool isenum=false, int* oListLen=nullptr) {
		char p[XML_MAX_SECTION_DESC_LEN];
		char ps[XML_MAX_SECTION_DESC_LEN+2];
		char pdesc[XML_MAX_PARAM_NAME_LEN];
		char pval[XML_MAX_PARAM_VAL_LEN];

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

#endif

} tParmsSource;
*/