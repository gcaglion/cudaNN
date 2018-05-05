#pragma once
#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"

#define DEFAULT_DBG_FPATH "C:/temp/logs"
#define DEFAULT_DBG_FNAME "Debugger"
#define DEFAULT_DBG_VERBOSITY true
#define DEFAULT_DBG_TIMING false
#define DEFAULT_DBG_PAUSERR true
#define DBG_MSG_MAXLEN 1024
#define DBG_STACK_MAXLEN 32768

struct sDebuggerParms {
	bool verbose;
	bool timing;
	bool pauseOnError;

	EXPORT sDebuggerParms(bool verbose_=DEFAULT_DBG_VERBOSITY, bool timing_=DEFAULT_DBG_TIMING, bool pauseOnError_=DEFAULT_DBG_PAUSERR);
};

struct sDebugger {

	sDebuggerParms* parms;

	tFileInfo* outFile;
	char msg[DBG_MSG_MAXLEN]="";
	char stackmsg[DBG_STACK_MAXLEN]="";

	EXPORT sDebugger(char* outFileName=DEFAULT_DBG_FNAME, sDebuggerParms* parms_=nullptr, char* outFilePath=DEFAULT_DBG_FPATH);
	EXPORT ~sDebugger();

};
