#pragma once

#include "..\CommonEnv.h"
#include "..\fileInfo.h"
#include <stdio.h>
#include <time.h>

//-- DBG Levels. Sets the level for the specific tDebugInfo* object
#define DBG_LEVEL_ERR 0	//-- errors
#define DBG_LEVEL_STD 1	//-- standard
#define DBG_LEVEL_DET 2	//-- detailed

//-- DBG Destinations. Sets the destination for the specific tDebugInfo* object
#define DBG_DEST_SCREEN 0
#define DBG_DEST_FILE 1
#define DBG_DEST_BOTH 2

//-- default values for the two above
#define DBG_LEVEL_DEFAULT DBG_LEVEL_STD
#define DBG_DEST_DEFAULT DBG_DEST_SCREEN

typedef struct sDbg {
	int level;
	int dest;
	tFileInfo* outFile;
	bool PauseOnError;

	//-- concurrency stuff
	bool ThreadSafeLogging;
	HANDLE Mtx;

	//-- timing stuff
	bool timing;
	DWORD startTime;

	char errmsg[1024];
	char stackmsg[32768];

#ifdef __cplusplus
	//-- constructors
	EXPORT sDbg(int level_=DBG_LEVEL_DEFAULT, int dest_=DBG_DEST_DEFAULT, tFileInfo* outFile_=nullptr, bool timing_=false, bool PauseOnError_=true, bool ThreadSafeLogging_=false);
	~sDbg() {}

	EXPORT void write(int cat, const char* msg, int argcount, ...);
	EXPORT void compose(const char* msg, int argcount, ...);
private:
	template <typename T> void argOut(int msgType, char* submsg, T arg){
		if (msgType==DBG_LEVEL_ERR) {
			//-- file log is mandatory in case of error
			fprintf(outFile->handle, submsg, arg);
			//-- then, screen log only if defined by dest
			if (dest==DBG_DEST_SCREEN||dest==DBG_DEST_BOTH) printf(submsg, arg);
		} else {
			//-- check dest only
			if (dest==DBG_DEST_SCREEN||dest==DBG_DEST_BOTH) printf(submsg, arg);
			if (dest==DBG_DEST_FILE||dest==DBG_DEST_BOTH) fprintf(outFile->handle, submsg, arg);
		}
	}

#endif

} tDebugInfo;

#define safeCallE(DBG, desc, block) \
if(DBG->timing) DBG->startTime=timeGetTime(); \
DBG->write(DBG_LEVEL_STD, "%s\n", 1, desc); \
try {block;} catch (const char* e) { \
	DBG->write(DBG_LEVEL_ERR, "%s failed. Exception %s \n", 2, desc, e); \
	return -1; \
} \
if(DBG->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DBG->startTime));

#define safeCallR(DBG, desc, block) \
if(DBG->timing) DBG->startTime=timeGetTime(); \
DBG->write(DBG_LEVEL_STD, "%s\n", 1, desc); \
if((block)!=0){\
	DBG->write(DBG_LEVEL_ERR, "%s failed. \n", 1, desc); \
	return -1; \
} else{\
if(DBG->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DBG->startTime));\
}

#define FailWithMsg(msg) {}

#define safeCallTop(dbg, desc, call) dbg->write(DBG_LEVEL_STD, "%s\n", 1, desc); try{ call; } catch(std::exception e) { sprintf_s(dbg->stackmsg, "%s\n%s(): Error %d at line %d", dbg->stackmsg, __func__, errno, __LINE__); return -1; }
#define safeCall(dbg, call) try{ call; } catch(std::exception e) { sprintf_s(dbg->stackmsg, "%s\n%s(): Error %d at line %d", dbg->stackmsg, __func__, errno, __LINE__); throw std::runtime_error(dbg->stackmsg); }
#define bottomThrow(dbg)  sprintf_s(dbg->stackmsg, "%s\n%s(): Error %d at line %d", dbg->stackmsg, __func__, errno, __LINE__); throw std::runtime_error(dbg->stackmsg);