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

	sXMLelement();
	~sXMLelement();

	EXPORT void setFromDesc(const char* desc, bool fromRoot=false);
	void copyTo(sXMLelement* destElement);
	void appendTo(sXMLelement* destParm);

} tXMLelement;

typedef struct sKey : sXMLelement {
	char vLine[XML_MAX_SECTION_DESC_LEN+3];
	char keyStepTagStart[XML_MAX_SECTION_DESC_LEN+2];
	char keyStepTagEnd[XML_MAX_SECTION_DESC_LEN+3];

	bool find(tFileInfo* parmsFile);

} tKey;
typedef struct sParm : sXMLelement {
	char name[XML_MAX_PARAM_NAME_LEN];
	char val[XML_MAX_PARAM_VAL_LEN];

	EXPORT bool find(tFileInfo* parmsFile);
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

	char** pArrDesc;

	EXPORT sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_=nullptr);
	void newDebugger(tDebugger* dbg_);

	EXPORT bool gotoKey(char* soughtKeyDesc, bool fromRoot=false, bool ignoreError=false);


	template <typename T> EXPORT void get(T* opVal, char* soughtParmDesc, bool fromRoot=false, bool isenum=false, int* oListLen=nullptr) {

		//-- set current key & filePos
		soughtParm->setFromDesc(soughtParmDesc, fromRoot);

		//-- if absolute path, go to file start 
		if (fromRoot) rewind(parmsFile->handle);

		//-- if found key, lookup parm within key
		if(soughtParm->find(parmsFile)) {
			get_(soughtParmDesc, opVal, isenum, oListLen);
		}



	}

	//-- specific, single value: int(with or without enums), numtype, char*
	EXPORT void get_(char* pvalS, int* oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, bool* oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, char* oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, numtype* oparamVal, bool isenum, int* oListLen);
	//-- specific, arrays: int(with or without enums), numtype, char*
	EXPORT void get_(char* pvalS, int** oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, bool** oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, numtype** oparamVal, bool isenum, int* oListLen);
	EXPORT void get_(char* pvalS, char** oparamVal, bool isenum, int* oListLen);

} tParmsSource;

/*

*/