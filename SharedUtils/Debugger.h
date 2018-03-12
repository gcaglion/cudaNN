#pragma once

#include "..\CommonEnv.h"
#include "Generic.h"
#include "FileInfo.h"
#include <stdio.h>
#include <time.h>
#include "Debugger_enums.h"

typedef struct sDebugger {
	int level;	// DBG_LEVEL_ERR ||DBG_LEVEL_STD || DBG_LEVEL_DET
	int dest;	// DBG_DEST_SCREEN || DBG_DEST_FILE || DBG_DEST_BOTH
	tFileInfo* outFile;
	bool PauseOnError;
	char errmsg[1024];

	//-- concurrency stuff
	bool ThreadSafeLogging;
	HANDLE Mtx;

	//-- timing stuff
	bool timing;
	DWORD startTime;
	DWORD elapsedTime;

#ifdef __cplusplus
	//-- constructor (fully defaulted)
	EXPORT sDebugger(int level_=DBG_LEVEL_DEFAULT, int dest_=DBG_DEST_DEFAULT, tFileInfo* outFile_=nullptr, bool timing_=false, bool PauseOnError_=true, bool ThreadSafeLogging_=false);
	EXPORT ~sDebugger();

	EXPORT void write(int LogType, const char* msg, int argcount, ...);
	EXPORT void compose(char* mask, int argcount, ...);	//-- writes resulting message into errmsg

	EXPORT void setStartTime();
	EXPORT void setElapsedTime();

private:
	template <typename T> void argOut(int msgType, char* submsg, T arg);

#endif

} tDebugger;

//-- 0. all messages need to be parametric

//-- caller AND callee can be either: 
//--							- B (Boolean, not throwing exception)
//--							or
//--							- C (Class,   throwing exception)
//-- therefore, we need:
//--	safeCallBB()
//--	safeCallEE()
//--	safeCallBE()
//--	safeCallEB()

//-- main debugger declaration & creation
#define createMainDebugger(level, dest) \
sDebugger* dbg=nullptr; \
try { \
	dbg=new tDebugger(level, dest, new tFileInfo("mainDebugger.log", DEBUG_DEFAULT_PATH)); \
} \
catch (std::exception e) { \
	if (dbg==nullptr) { \
		fprintf(stderr, "\n CRITICAL ERROR: could not create main client debugger!\n"); \
		system("pause"); return -1; \
	} \
}

//-- class calling class
#define safeCallEE(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s ... ", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (std::exception e) { \
		dbg->write(DBG_LEVEL_STD, "FAILURE!\n", 0); \
		throw(e); \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
//-- class calling boolean
#define safeCallEB(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s ... ", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	if(!(block)){ \
		dbg->write(DBG_LEVEL_STD, "FAILURE!\n", 0); \
		throw std::exception("Call to bool function failed"); \
	} else {\
		dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	}\
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
//-- boolean calling class
#define safeCallBE(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s ... ", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (std::exception e) { \
		dbg->write(DBG_LEVEL_STD, "FAILURE!\n", 0); \
		return false; \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
//-- boolean calling boolean
#define safeCallBB(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s\n", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	if(!(block)) {\
		dbg->write(DBG_LEVEL_STD, "FAILURE!\n", 0); \
		return false; \
	} else { \
		dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	} \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}

//-- throw exception from class method
#define throwE(mask, argcnt, ...) { \
	dbg->compose((#mask), argcnt, __VA_ARGS__ ); \
	dbg->write(DBG_LEVEL_ERR, "%s() failed with message: %s \n", 2, __func__, dbg->errmsg); \
	throw std::exception(dbg->errmsg); \
}
//-- return error from boolean function
#define throwB(mask, argcnt, ...) { \
	dbg->compose((#mask), argcnt, __VA_ARGS__ ); \
	dbg->write(DBG_LEVEL_ERR, "%s() failed with message: %s \n", 2, __func__, dbg->errmsg); \
	return false; \
}


//-- case 3a: throw exception from Debugger constructor
//-- case 3b: caller of Debugger constructor