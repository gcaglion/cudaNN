#pragma once
#include "CommonEnv.h"
#include <stdio.h>


typedef struct sFileInfo {
	char Path[MAX_PATH];
	char Name[MAX_PATH];
	char FullName[MAX_PATH];
	FILE* handle;
	char creationTime[13];
	bool append;

#ifdef __cplusplus

	sFileInfo(char* Name_, char* Path_=DEBUG_DEFAULT_PATH, bool append_=false) {
		strcpy_s(Name, MAX_PATH, Name_);
		strcpy_s(Path, MAX_PATH, Path_);
		sprintf_s(creationTime, sizeof(creationTime), "%ld", timeGetTime());
		sprintf_s(FullName, MAX_PATH-1, "%s/%s_%s.log", Path, Name, creationTime);

		fopen_s(&handle, FullName, (append) ? "a" : "w");
		if (errno!=0) {
			sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d creating file %s", __func__, errno, FullName); throw std::runtime_error(errmsg);
		}
	}

	~sFileInfo() {
		fclose(handle);
	}

private:
	char errmsg[1024]; 

#endif

} tFileInfo;