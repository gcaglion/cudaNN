#pragma once

#include "../CommonEnv.h"
#include "../DBConnection/DBConnection.h"
#include "../DataSource/FileData.h"
#include "../ParamMgr/ParamMgr.h"
#include "../MyAlgebra/MyAlgebra.h"	//-- we need this in LoadW() and SaveW()
#include "Logger_enums.h"

#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif

typedef struct sLogger : public sBaseObj {
	int dest;
	tAlgebra* Alg;	//-- we need this in LoadW() and SaveW()
	tDBConnection* db=nullptr;
	tFileData* file=nullptr;
	bool saveNothing;
	bool saveClient;
	bool saveMSE;
	bool saveRun;
	bool saveInternals;
	bool saveImage;

	EXPORT sLogger(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sLogger(char* objName_, sBaseObj* objParent_, tDBConnection* logDB, bool saveNothing_=false, bool saveClient_=true, bool saveMSE_=true, bool saveRun_=true, bool saveInternals_=false, bool saveImage_=true, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sLogger(char* objName_, sBaseObj* objParent_, tFileData* logFile, sDebuggerParms* dbgparms_=nullptr);

	EXPORT void SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
	EXPORT void SaveRun(int pid, int tid, int setid, int npid, int ntid, int runCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual);
	EXPORT void SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT void LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT void SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTrain, bool doTrainRun, bool doTestRun);
	EXPORT void Commit();

} tLogger;