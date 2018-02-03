#include "Logger.h"

EXPORT int LogSaveMSE(tDebugInfo* DebugParms, int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV) {
	if (DebugParms->DebugDest==LOG_TO_ORCL) {
		return (Ora_LogSaveMSE(DebugParms, pid, tid, mseCnt, mseT, mseV));
	} else {
		return 0;
	}
}
EXPORT int LogSaveRun(tDebugInfo* DebugParms, int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual) {
	if (DebugParms->DebugDest==LOG_TO_ORCL) {
		return (Ora_LogSaveRun(DebugParms, pid, tid, runCnt, featuresCnt, prediction, actual));
	} else {
		return 0;
	}
	}
EXPORT int LogSaveW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W) {
	numtype* hW;
	#ifdef USE_GPU
	hW=(numtype*)malloc(Wcnt*sizeof(numtype));
	if (cudaMemcpy(hW, W, Wcnt*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	#else
		hW=W;
	#endif
	if (DebugParms->DebugDest==LOG_TO_ORCL) {
		return (Ora_LogSaveW(DebugParms, pid, tid, epoch, Wcnt, hW));
	} else {
		return 0;
	}
	#ifdef USE_GPU
	free(hW);
	#endif
}
EXPORT int LogLoadW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W) {
	numtype* hW;
#ifdef USE_GPU
	hW=(numtype*)malloc(Wcnt*sizeof(numtype));
#else
	hW=W;
#endif
	if (DebugParms->DebugDest==LOG_TO_ORCL) {
		return (Ora_LogLoadW(DebugParms, pid, tid, epoch, Wcnt, hW));
	} else {
		return 0;
	}
#ifdef USE_GPU
	if (cudaMemcpy(W, hW, Wcnt*sizeof(numtype), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
	free(hW);
#endif
}

EXPORT void Commit(tDebugInfo* DebugParms) {
	if (DebugParms->DebugDest==LOG_TO_ORCL) {
		OraCommit(DebugParms->DebugDB->DBCtx);
	} else {
		fclose(DebugParms->fHandle);
	}
}