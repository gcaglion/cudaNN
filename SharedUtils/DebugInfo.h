#pragma once

#include "..\CommonEnv.h"
#include "FileInfo.h"
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

	EXPORT void write(int LogType, const char* msg, int argcount, ...);
	EXPORT void compose(char* mask, int argcount, ...);

private:
	template <typename T> void argOut(int msgType, char* submsg, T arg);

#endif

} tDebugInfo;
/*
#define safeCallE(desc, block) \
if(DebugParms->timing) DebugParms->startTime=timeGetTime(); \
DebugParms->write(DBG_LEVEL_STD, "%s\n", 1, desc); \
try {block;} catch (std::exception e) { \
	DebugParms->write(DBG_LEVEL_ERR, "%s failed. Exception %s \n", 2, desc, e.what()); \
	return -1; \
} \
if(DebugParms->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DebugParms->startTime));
*/
#define safeCallR(desc, block) \
if(DebugParms->timing) DebugParms->startTime=timeGetTime(); \
DebugParms->write(DBG_LEVEL_STD, "%s\n", 1, desc); \
if((block)!=0){\
	DebugParms->write(DBG_LEVEL_ERR, "%s failed. \n", 1, desc); \
	return -1; \
} else{\
if(DebugParms->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DebugParms->startTime));\
}


//#define safeCallTop(desc, call) DebugParms->write(DBG_LEVEL_STD, "%s\n", 1, desc); try{ call; } catch(std::exception e) { sprintf_s(DebugParms->stackmsg, "%s\n%s(): Error %d at line %d", DebugParms->stackmsg, __func__, errno, __LINE__); return -1; }

#define safeCall_(desc, block, arg3, arg4) \
if(DebugParms->timing) DebugParms->startTime=timeGetTime(); \
DebugParms->write(DBG_LEVEL_STD, "%s\n", 1, desc); \
try {block;} catch (std::exception e) { \
	DebugParms->write(DBG_LEVEL_ERR, "%s failed. Exception %s \n", 2, desc, e.what()); \
	return -1; \
} \
if(DebugParms->timing) printf("%s : elapsed time=%ld \n", desc, (DWORD)(timeGetTime()-DebugParms->startTime));

#define bottomThrow(mask, varcnt, ...) {\
sprintf_s(DebugParms->stackmsg, "%s\n%s(): Error %d at line %d: ", DebugParms->stackmsg, __func__, errno, __LINE__); \
DebugParms->compose(mask, varcnt, __VA_ARGS__); \
throw std::runtime_error(DebugParms->stackmsg);\
}

/*
#define safeCall_1_args(call) safeCall_(call, 0, 0, 0)
#define safeCall_2_args(desc, call) safeCall_(desc, call, 0, 0)
#define safeCall_3_args(desc, call, arg3) safeCall_(desc, call, arg3)
#define safeCall_4_args(desc, call, arg3, arg4) safeCall_(desc, call, arg3, arg4)

#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define safeCall_Pick(...) GET_4TH_ARG(__VA_ARGS__, safeCall_3_args, safeCall_2_args, safeCall_1_args, )

#define safeCall(...) safeCall_Pick(__VA_ARGS__)(__VA_ARGS__)
*/
#define safeCall_1_args(call) safeCall_(call,0 , 0, 0)
#define safeCall_2_args(desc, call) safeCall_(desc, call, 0, 0)
#define safeCall_3_args(desc, call, arg3) safeCall_(desc, call, arg3, 0)
#define safeCall_4_args(desc, call, arg3, arg4) safeCall_(desc, call, arg3, arg4)

#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define safeCall_Pick(...) GET_4TH_ARG(__VA_ARGS__, safeCall_3_args, safeCall_2_args, safeCall_1_args, )

#define safeCall(...) safeCall_Pick(__VA_ARGS__)(__VA_ARGS__)

