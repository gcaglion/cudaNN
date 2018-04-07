#pragma once
#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"
#include "../Debugger/Debugger.h"
#include "../Utils/Utils.h"
#include "ParamMgr_limits.h"

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
	char bkpKey[XML_MAX_PATH_LEN];

	char soughtKey[XML_MAX_PATH_LEN];
	char soughtParm[XML_MAX_PARAM_NAME_LEN];
	char soughtParmFull[XML_MAX_PATH_LEN];

	//--
	EXPORT bool setKey(char* KeyDesc, bool ignoreError=false);
	EXPORT bool backupKey();
	EXPORT bool restoreKey();
	bool findKey(char* KeyFullDesc);

	EXPORT bool parse();
	//--
	EXPORT bool decode(int elementId, int* oVal);

	//-- we need this in .cpp because it uses UpperCase() from Utils lib, which is not linked by all modules calling ParamMgr...
	EXPORT void buildSoughtParmFull(const char* soughtParmDesc);
	EXPORT int findParmId();
	//--
	template <typename T> EXPORT void get(T* oVar, const char* soughtParmDesc, int* oListLen=nullptr) {

		buildSoughtParmFull(soughtParmDesc);

		//-- lookup parm name&val
		foundParmId=findParmId();
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
