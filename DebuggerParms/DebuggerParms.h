#pragma once
#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"
#include "Debugger_enums.h"

#define DEFAULT_DBG_DEST		DBG_DEST_BOTH
#define DEFAULT_DBG_FPATH		"C:/temp/logs"
#define DEFAULT_DBG_FNAME		"Debugger"
#define DEFAULT_DBG_VERBOSITY	true
#define DEFAULT_DBG_TIMING		false
#define DEFAULT_DBG_PAUSERR		true

#define DBG_MSG_MAXLEN			1024
#define DBG_STACK_MAXLEN		32768

typedef struct sDebuggerParms {
	int dest;
	bool verbose;
	bool timing;
	bool pauseOnError;
	char outFilePath[MAX_PATH];
	char outFileName[MAX_PATH];
	char outFileFullName[MAX_PATH];
	//FILE* outFileHandle;
	tFileInfo* outFile;

#ifdef __cplusplus
	EXPORT sDebuggerParms(char* ownerObjName_=nullptr, int dest_=DEFAULT_DBG_DEST, bool verbose_=DEFAULT_DBG_VERBOSITY, bool timing_=DEFAULT_DBG_TIMING, bool pauseOnError_=DEFAULT_DBG_PAUSERR, char* outFileFullName_=nullptr, char* outFilePath_=nullptr);
	EXPORT ~sDebuggerParms();
#endif
} tDebuggerParms;

