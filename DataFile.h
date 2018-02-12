// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"
#include "fileInfo.h"

#define MAXDATASETS 8
typedef struct sDataFile {
	tFileInfo* file;
	int datasetsCnt;
	int fieldSep;	// COMMA|TAB|SPACE
	int dataSet[MAXDATASETS];
	int calcBW;
	int BWDataSet[MAXDATASETS];
} tDataFile;
