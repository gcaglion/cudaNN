#pragma once

#include "..\CommonEnv.h"
#include "..\DBConnection.h"
#include <stdio.h>
#include <time.h>


// Log Level
#define LOG_INFO 0
#define LOG_ERROR 1

// Logs Destinations
#define LOG_TO_TEXT   1
#define LOG_TO_ORCL	  2

typedef struct sDebugInfo {
	int DebugLevel;		//-- 0:Nothing ; 1:Screen-Only ; 2:File-Only ; 3:File+Screen
	int DebugDest;		//-- ORCL | TEXT
	int PauseOnError;
	tDBConnection* DebugDB;
	char fPath[MAX_PATH];
	char fName[MAX_PATH];
	char FullfName[MAX_PATH];
	FILE* fHandle;
	int  fIsOpen;
	void* DBCtx;
	int ThreadSafeLogging;
	int SaveNothing;
	int SaveMSE;
	int SaveRun;
	int SaveInternals;
	int SaveImages;
	int DumpSampleData;
	HANDLE Mtx;		// Mutex handle used by LogWrite()
#ifdef __cplusplus
	sDebugInfo() {
		DebugDB = new tDBConnection();
	}

	~sDebugInfo() {
		delete(DebugDB);
	}
#endif
} tDebugInfo;

#ifdef __cplusplus
EXPORT void LogWrite(tDebugInfo* DebugParms, int LogType, const char* msg, int argcount, ...);
#endif
