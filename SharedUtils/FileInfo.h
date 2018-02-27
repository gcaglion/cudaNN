#pragma once
#include "../CommonEnv.h"
#include <stdio.h>

#define FILE_MODE_READ   0
#define FILE_MODE_WRITE  1
#define FILE_MODE_APPEND 2

typedef struct sFileInfo {
	char Path[MAX_PATH];
	char Name[MAX_PATH];
	char FullName[MAX_PATH];
	FILE* handle;
	char creationTime[13];
	int mode;
	char modeS[2];
	char modeDesc[30];

#ifdef __cplusplus
	EXPORT sFileInfo(char* Name_, char* Path_=DEBUG_DEFAULT_PATH, int mode_=FILE_MODE_WRITE);
	EXPORT sFileInfo(char* FullName_, int mode_);
	EXPORT ~sFileInfo();

private:
	char errmsg[1024]; 
	void setModeS(int mode_);
#endif

} tFileInfo;