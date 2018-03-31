#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "Generic.h"

#define XML_MAX_PARAMS_CNT 32768
#define XML_MAX_PATH_DEPTH 128
#define XML_MAX_SECTION_DESC_LEN 64
#define XML_MAX_PATH_LEN XML_MAX_PATH_DEPTH*XML_MAX_SECTION_DESC_LEN
#define XML_MAX_PARAM_NAME_LEN	128
#define XML_MAX_PARAM_VAL_LEN	128
#define XML_MAX_LINE_SIZE XML_MAX_PARAM_NAME_LEN+XML_MAX_PARAM_VAL_LEN
#define XML_MAX_ARRAY_PARAM_ELEM_CNT 32

//-- get options
#define SimpleVal 8
#define EnumVal   9

typedef struct sParmsSource {

	tDebugger* dbg;

	tFileInfo* parmsFile;
	int CLoverridesCnt;
	char** CLoverride;

	int parmsCnt;
	int foundParmId;

	EXPORT sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_=nullptr);
	EXPORT ~sParmsSource();
	void newDebugger(tDebugger* dbg_);

	char parmName[XML_MAX_PARAMS_CNT][XML_MAX_PATH_LEN];
	
	int parmValsCnt[XML_MAX_PARAMS_CNT];
	char*** parmVal;	// [XML_MAX_PARAMS_CNT][XML_MAX_ARRAY_PARAM_ELEM_CNT][XML_MAX_PARAM_VAL_LEN];

	char currentKey[XML_MAX_PATH_LEN];

	char soughtKey[XML_MAX_PATH_LEN];
	char soughtParm[XML_MAX_PARAM_NAME_LEN];
	char soughtParmFull[XML_MAX_PATH_LEN];

	//--
	EXPORT bool setKey(char* KeyDesc, bool ignoreError=false);
	bool findKey(char* KeyFullDesc);

	EXPORT bool parse();
	//--
	EXPORT bool decode(int elementId, int* oVal);

	template <typename T> EXPORT void get(T* oVar, const char* soughtParmDesc, int* oListLen=nullptr) {

		if (soughtParmDesc[0]=='.' || strlen(currentKey)==0) {
			soughtKey[0]='\0';
		} else {
			strcpy_s(soughtKey, XML_MAX_PATH_LEN, currentKey);
			strcat_s(soughtKey, ".");
		}
		strcpy_s(soughtParmFull, XML_MAX_PATH_LEN, soughtKey);
		strcat_s(soughtParmFull, soughtParmDesc);
		UpperCase(soughtParmFull);

		//-- lookup parm name&val
		foundParmId=-1;
		for (int p=0; p<parmsCnt; p++) {
			if (strcmp(soughtParmFull, parmName[p])==0) {
				foundParmId=p;
				break;
			}
		}
		if (foundParmId<0) throwE("could not find parm %s. currentKey=%s", 1, soughtParmDesc, currentKey);

		//-- set oListLen (if passed)
		if (oListLen!=nullptr) (*oListLen)=parmValsCnt[foundParmId];
		//-- call type-specific getx(), which in turn uses Fully-qualified parameter name
		getx(oVar);

	}

	//-- type-specific: int(with or without enums), numtype, char*
	EXPORT void getx(int* oVar);
	EXPORT void getx(bool* oVar);
	EXPORT void getx(char* oVar);
	EXPORT void getx(numtype* oVar);
	//-- type-specific, arrays: int(with or without enums), numtype, char*
	EXPORT void getx(int** oVar);
	EXPORT void getx(bool** oVar);
	EXPORT void getx(char** oVar);
	EXPORT void getx(numtype** oVar);

} tParmsSource;
