#pragma once
#include "CommonEnv.h"
#include <stdio.h>

#define FILEINFO_CREATION_FAILED(fname) "Failed to open file ((fname)))"

typedef struct sFileInfo {
	char Path[MAX_PATH];
	char Name[MAX_PATH];
	char FullName[MAX_PATH];
	FILE* handle;
	bool append;

#ifdef __cplusplus

	sFileInfo(char* Path_, char* Name_, bool append_=false) {
		strcpy_s(Name, MAX_PATH, Name_);
		if (Path_==nullptr) {
			strcpy_s(Path, MAX_PATH, DEBUG_DEFAULT_PATH);
		} else {
			strcpy_s(Path, MAX_PATH, Path_);
		}
		sprintf_s(FullName, MAX_PATH, "%s/%s", Path, Name);

		fopen_s(&handle, FullName, (append) ? "a" : "w");
		if (errno!=0) throw FILEINFO_CREATION_FAILED(FullName);
	}

	~sFileInfo() {
		fclose(handle);
	}

#endif

} tFileInfo;