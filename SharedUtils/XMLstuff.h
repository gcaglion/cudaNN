#pragma once
#include "../CommonEnv.h"
#include "Generic.h"
#include "FileInfo.h"

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
	EXPORT bool decode();
} tParm;

