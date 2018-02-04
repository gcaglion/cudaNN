#pragma once

#include "../CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../OraUtils/OraUtils.h"
#include "../DBConnection.h"
#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

EXPORT int LogSaveMSE(tDebugInfo* DebugParms, int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
EXPORT int LogSaveRun(tDebugInfo* DebugParms, int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual);
EXPORT int LogSaveW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT int LogLoadW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT int LogSaveClient(tDebugInfo* DebugParms, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun);

EXPORT void Commit(tDebugInfo* DebugParms);
