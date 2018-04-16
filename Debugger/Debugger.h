#pragma once

#include "../CommonEnv10.h"

#include <stdio.h>
#include <time.h>
#include "../Utils/Utils.h"
#include "../FileInfo/FileInfo.h"
#include "Debugger_enums.h"

//-- defaults
#define DBG_DEFAULT_PATH "C:/temp/logs"
#define DBG_DEFAULT_NAME "Debugger"
#define DBG_DEFAULT_LEVEL DBG_LEVEL_ERR
#define DBG_DEFAULT_DEST DBG_DEST_BOTH
#define DBG_ERRMSG_SIZE 32768

typedef struct sDebugger
#ifdef __cplusplus
	: sBaseObj10
#endif
{
	char parentObjName[MAX_PATH]
#ifdef __cplusplus
	=""
#endif
	;
	int level;	// DBG_LEVEL_ERR ||DBG_LEVEL_STD || DBG_LEVEL_DET
	int dest;	// DBG_DEST_SCREEN || DBG_DEST_FILE || DBG_DEST_BOTH
	tFileInfo* outFile
#ifdef __cplusplus
		=nullptr
#endif
	;
	char errmsg[DBG_ERRMSG_SIZE]
#ifdef __cplusplus
		=""
#endif
	;
	bool PauseOnError;

	//-- concurrency stuff
	bool ThreadSafeLogging;
	HANDLE Mtx;

	//-- timing stuff
	bool timing;
	DWORD startTime;
	DWORD elapsedTime;

#ifdef __cplusplus
	EXPORT void sDebugger_common(int level_, int dest_, char* outFileName_, char* outFilePath_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_);
	EXPORT sDebugger(int level_, int dest_, char* outFileName_, char* outFilePath_, bool timing_=false, bool PauseOnError_=false, bool ThreadSafeLogging_=false);
	EXPORT sDebugger(char* outFileName_);
	EXPORT sDebugger(int level_, int dest_, char* objName_);
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
//--	safeCall()
//--	safeCallBE()
//--	safeCall()

//-- main debugger declaration & creation
#define createMainDebugger(level, dest) \
sDebugger* dbg=nullptr; \
DWORD mainStart=timeGetTime(); \
try { \
	dbg=new tDebugger(level, dest, "mainDebugger"); \
} \
catch (char* e) { \
	if (dbg==nullptr) { \
		fprintf(stderr, "\n CRITICAL ERROR: could not create main client debugger! Exception: %s\n", e); \
		system("pause"); return -1; \
	} \
}

//-- class calling class
/*#define safeCallEE(block) { \
	dbg->write(DBG_LEVEL_STD, "%s -> %s() calling %s ... ", 3, dbg->parentObjName, __func__, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (char* e) { \
		dbg->write(DBG_LEVEL_ERR, "%s -> %s() FAILURE: %s\n", 3, dbg->parentObjName, __func__, e); \
		throw(e); \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}*/
#define safeCallEE(block) { \
	dbg->write(DBG_LEVEL_STD, "%s -> %s() calling %s ... ", 3, dbg->parentObjName, __func__, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (char* exc) { \
		printf("exc=%s\n", exc); \
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s -> %s() FAILURE: %s\n",dbg->parentObjName, __func__, exc); \
		throw(dbg->errmsg); \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}

//-- class calling boolean
#define safeCallEB(block) { \
	dbg->write(DBG_LEVEL_STD, "%s -> %s() calling %s ... ", 3, dbg->parentObjName, __func__, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	if(!(block)){ \
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE-1, "%s -> %s() FAILURE!\n",dbg->parentObjName, __func__); \
		throw(dbg->errmsg); \
	} else {\
		dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	}\
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
//-- boolean calling class
/*#define safeCallBE(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s ... ", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (std::exception e) { \
		dbg->write(DBG_LEVEL_ERR, "dbg %p FAILURE: %s\n", 2, dbg, e.what()); \
		return false; \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
*/
//-- boolean calling boolean
/*
#define safeCallBB(block) { \
	dbg->write(DBG_LEVEL_STD, "calling %s\n", 1, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	if(!(block)) {\
		dbg->write(DBG_LEVEL_ERR, "FAILURE!\n", 0); \
		return false; \
	} else { \
		dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	} \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}
*/
//-- throw exception from class method
#define throwE(mask, argcnt, ...) { \
	dbg->compose((#mask), argcnt, __VA_ARGS__ ); \
	dbg->write(DBG_LEVEL_ERR, "%s -> %s() failed with message: %s \n", 3, dbg->parentObjName, __func__, dbg->errmsg); \
	throw dbg->errmsg; \
}
//-- return error from boolean function
#define throwB(mask, argcnt, ...) { \
	dbg->compose((#mask), argcnt, __VA_ARGS__ ); \
	dbg->write(DBG_LEVEL_ERR, "%s -> %s() failed with message: %s \n", 3, dbg->parentObjName, __func__, dbg->errmsg); \
	return false; \
}



//==========================
// on failure, functions can either throw an exception, OR return false. Never both
// 

#define safeThrow(mask, argcnt, ...) { \
	dbg->compose((#mask), argcnt, __VA_ARGS__ ); \
	printf("dbg->parentObjName=%s\n", dbg->parentObjName); \
	printf("dbg->errmsg=%s\n", dbg->errmsg); \
	dbg->compose("safeThrow(): %s -> %s() failed with message: %s \n", 3, dbg->parentObjName, __func__, dbg->errmsg); \
	throw dbg->errmsg; \
}

#define safeCall(block) { \
	dbg->write(DBG_LEVEL_STD, "%s -> %s() calling %s ... ", 3, dbg->parentObjName, __func__, (#block)); \
	if(dbg->timing) dbg->setStartTime(); \
	try {block;} catch (char* exc) { \
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s -> %s() FAILURE: %s\n",dbg->parentObjName, __func__, exc); \
		printf("DioPorco! dbg->errmsg=%s\n", dbg->errmsg); \
		throw(dbg->errmsg); \
	} \
	dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0); \
	if(dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); } \
	dbg->write(DBG_LEVEL_STD, "\n", 0); \
}

//==========================
