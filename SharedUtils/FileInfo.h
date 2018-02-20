#pragma once
#include "../CommonEnv.h"
#include <stdio.h>


typedef struct sFileInfo {
	char Path[MAX_PATH];
	char Name[MAX_PATH];
	char FullName[MAX_PATH];
	FILE* handle;
	char creationTime[13];
	bool append;

#ifdef __cplusplus

	EXPORT sFileInfo(char* Name_, char* Path_=DEBUG_DEFAULT_PATH, bool append_=false);
	~sFileInfo();
private:
	char errmsg[1024]; 

#endif

} tFileInfo;