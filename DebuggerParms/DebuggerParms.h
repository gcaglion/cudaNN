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

typedef struct sDebuggerC {
	char* name;
	int stackLevel;
	void* parent;

	int dest;
	bool verbose;
	bool timing;
	bool pauseOnError;
	/*char outFilePath[MAX_PATH];
	char outFileName[MAX_PATH];
	char outFileFullName[MAX_PATH];*/
	tFileInfo* outFile;

	char* dbgmsgmask;
	char* dbgmsg;
	char* stackmsg;
} tDebuggerC;

typedef struct sDebuggerParms {
	int dest;
	bool verbose;
	bool timing;
	bool pauseOnError;
	char outFilePath[MAX_PATH];
	char outFileName[MAX_PATH];
	char outFileFullName[MAX_PATH];
	tFileInfo* outFile;

	tDebuggerC* Cdbg;

#ifdef __cplusplus
	EXPORT sDebuggerParms(char* ownerObjName_=nullptr, int dest_=DEFAULT_DBG_DEST, bool verbose_=DEFAULT_DBG_VERBOSITY, bool timing_=DEFAULT_DBG_TIMING, bool pauseOnError_=DEFAULT_DBG_PAUSERR, char* outFileFullName_=nullptr, char* outFilePath_=nullptr);
	EXPORT ~sDebuggerParms();
#endif
} tDebuggerParms;



#define setmsgC(mask, ...){ \
		sprintf_s(dbg->dbgmsgmask, DBG_MSG_MAXLEN, "%s(%p)->%s() %s:", dbg->name, "Cdbg", __func__, mask); \
		sprintf_s(dbg->dbgmsg, DBG_MSG_MAXLEN, dbg->dbgmsgmask, __VA_ARGS__); \
}
#define infoC(mask, ...)  { \
	if(dbg->verbose) { \
		setmsgC(mask, __VA_ARGS__); \
		_foutC(true); \
	} \
}

#define errC(mask, ...) { \
	setmsgC(mask, __VA_ARGS__); \
	_foutC(false); \
}

//EXPORT void _foutC(tDebuggerC* dbg, Bool success);
#define _foutC(success) { \
	for (int t=0; t<dbg->stackLevel; t++) sprintf_s(dbg->dbgmsg, DBG_MSG_MAXLEN, "\t%s", dbg->dbgmsg); \
	strcat_s(dbg->dbgmsg, DBG_MSG_MAXLEN, "\n"); \
	strcat_s(dbg->stackmsg, DBG_STACK_MAXLEN, dbg->dbgmsg); \
	printf("%s", dbg->dbgmsg); \
	if (dbg->outFile!=NULL) fprintf(dbg->outFile->handle, "%s", dbg->dbgmsg); \
	if (!success && dbg->pauseOnError) { printf("Press any key..."); getchar(); } \
}

//if (stackLevel>0) sprintf_s(parent->stackmsg, DBG_STACK_MAXLEN, "%st%s", dbg->parent->stackmsg, dbg->dbgmsg);
