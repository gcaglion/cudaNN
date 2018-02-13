#pragma once

#include "..\CommonEnv.h"
#include "..\fileInfo.h"
#include <stdio.h>
#include <time.h>


// Debug Message Types
#define DBG_INFO 0
#define DBG_ERROR 1

typedef struct sDebugInfo {
	int level;		//-- 0:Nothing ; 1:Screen-Only ; 2:File-Only ; 3:File+Screen
	tFileInfo* outFile;
	bool PauseOnError;
	bool ThreadSafeLogging;
	HANDLE Mtx;		// Mutex handle used by LogWrite()

	//--
	bool timing;
	DWORD startTime;
	//--

	//--
	//--

#ifdef __cplusplus

	EXPORT sDebugInfo(int level_, char* fName_, char* fPath_=DEBUG_DEFAULT_PATH, bool timing_=false, bool append_=false);
	EXPORT ~sDebugInfo();

	EXPORT void write(int LogType, const char* msg, int argcount, ...);

#endif

} tDebugInfo;

#ifdef __cplusplus
//EXPORT void LogWrite(tDebugInfo* DebugParms, int LogType, const char* msg, int argcount, ...);
#endif

#define safeCallE(desc, block) \
if(DBG->timing) DBG->startTime=timeGetTime(); \
try {block;} catch (const char* e) { \
	DBG->write(DBG_ERROR, "%s failed. Exception %s \n", 2, desc, e); \
	return -1; \
} \
if(DBG->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DBG->startTime));

#define safeCallR(desc, block) \
if(DBG->timing) DBG->startTime=timeGetTime(); \
if((block)!=0){\
	DBG->write(DBG_ERROR, "%s failed. \n", 1, desc); \
	return -1; \
} else{\
if(DBG->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DBG->startTime));\
}
