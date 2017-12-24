// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"

#define MAXDATASETS 8
typedef struct sFileData {
	char FileName[MAX_PATH];
	int FileDataSetsCount;
	int FieldSep;	// COMMA|TAB|SPACE
	int* FileDataSet;
	int CalcFileDataBW;
	int* FileBWDataSet;
#ifdef __cplusplus
	sFileData() {
		FileDataSet = (int*)malloc(MAXDATASETS*sizeof(int));
		FileBWDataSet = (int*)malloc(MAXDATASETS*sizeof(int));
	}

	~sFileData() {
		free(FileDataSet);
		free(FileBWDataSet);
	}
#endif
} tFileData;
