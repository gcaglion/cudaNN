#pragma once

#include "../CommonEnv00.h"

#include <stdio.h>
#include <time.h>
#include "../Utils/Utils.h"
#include "FileInfo_enums.h"


typedef struct sFileInfo {

	char objName[64]="";
	void* objParent=nullptr;

	char Path[MAX_PATH];
	char Name[MAX_PATH];
	char FullName[MAX_PATH];
	FILE* handle;
	fpos_t pos;

	int mode;
	char modeS[2];
	char modeDesc[30];

#ifdef __cplusplus
	EXPORT void sFileInfo_common();
	EXPORT sFileInfo(char* Name_, char* Path_, int mode_);
	EXPORT sFileInfo(char* FullName_, int mode_);
	EXPORT sFileInfo(char* objName_, void* objParent_, char* Name_, char* Path_, int mode_);
	EXPORT sFileInfo(char* objName_, void* objParent_, char* FullName_, int mode_);
	EXPORT ~sFileInfo();
	EXPORT void savePos();
	EXPORT void restorePos();

private:
	char errmsg[1024]; 
	void setModeS();
#endif

} tFileInfo;

#define spawnFile(file, fname, fmode){ \
	try { \
		file=new tFileInfo(fname, fmode); \
	} catch (std::exception exc) { \
		throw(exc); \
	} \
}
