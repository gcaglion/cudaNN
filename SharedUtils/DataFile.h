// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"
#include "FileInfo.h"

#define MAXDATASETS 8

//-- field separators
#define DATAFILE_SEP_COMMA 0
#define DATAFILE_SEP_TAB 1
#define DATAFILE_SEP_SPACE 2

typedef struct sDataFile {
	tFileInfo* file;
	int datasetsCnt;
	int fieldSep;
	int dataSet[MAXDATASETS];
	bool calcBW;
	int BWDataSet[MAXDATASETS];
} tDataFile;
