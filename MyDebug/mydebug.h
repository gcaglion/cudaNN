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
	int PauseOnError;
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

#define safeCallE(desc, debugParms, block) \
if(debugParms->timing) debugParms->startTime=timeGetTime(); \
try {block;} catch (const char* e) { \
	debugParms->write(DBG_ERROR, "%s failed. Exception %s \n", 2, desc, e); \
	return -1; \
} \
if(debugParms->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-debugParms->startTime));

#define safeCallR(desc, debugParms, block) \
if(debugParms->timing) debugParms->startTime=timeGetTime(); \
if((block)!=0){\
	debugParms->write(DBG_ERROR, "%s failed. \n", 1, desc); \
	return -1; \
} else{\
if(debugParms->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-debugParms->startTime));\
}
