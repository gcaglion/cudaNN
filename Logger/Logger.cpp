#include "Logger.h"

sLogger::sLogger(tDBConnection* logDB, tDebugInfo* DebugParms_) {
	if (DebugParms_==nullptr) {
		DebugParms=new tDebugInfo(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("sLogger.err"));
	} else {
		DebugParms=DebugParms_;
	}
	dest=LOG_TO_ORCL;
	db=logDB;
}
sLogger::sLogger(tDataFile* logFile) {
	dest=LOG_TO_TEXT;
	file=logFile;
}
sLogger::~sLogger() {}

bool sLogger::SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV) {
	if (saveMSE) {
		if (dest==LOG_TO_ORCL) {
			return (Ora_LogSaveMSE(DebugParms, db, pid, tid, mseCnt, mseT, mseV));
		} else {
			return false;
		}
	} else {
		return true;
	}
}
bool sLogger::SaveRun(int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual) {
	if (saveRun) {
		if (dest==LOG_TO_ORCL) {
			return (Ora_LogSaveRun(DebugParms, db, pid, tid, runCnt, featuresCnt, prediction, actual));
		} else {
			return false;
		}
	} else {
		return true;
	}
}

bool sLogger::SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W) {
	if (saveImage) {
		numtype* hW;
		#ifdef USE_GPU
			hW=(numtype*)malloc(Wcnt*sizeof(numtype));
			if (cudaMemcpy(hW, W, Wcnt*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return false;
		#else
			hW=W;
		#endif
		if (dest==LOG_TO_ORCL) {
			return (Ora_LogSaveW(DebugParms, db, pid, tid, epoch, Wcnt, hW));
		} else {
			return true;
		}
		
		#ifdef USE_GPU
			free(hW);
		#endif
	} else {
		return true;
	}
}
bool sLogger::LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W) {
	numtype* hW;
#ifdef USE_GPU
	hW=(numtype*)malloc(Wcnt*sizeof(numtype));
#else
	hW=W;
#endif
	if (dest==LOG_TO_ORCL) {
		return (Ora_LogLoadW(DebugParms, db, pid, tid, epoch, Wcnt, hW));
	} else {
		return false;
	}
#ifdef USE_GPU
	if (cudaMemcpy(W, hW, Wcnt*sizeof(numtype), cudaMemcpyHostToDevice)!=cudaSuccess) return false;
	free(hW);
#endif
}
bool sLogger::SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun) {
	if (saveClient) {
		if (dest==LOG_TO_ORCL) {
			return (Ora_LogSaveClient(DebugParms, db, pid, clientName, startTime, duration, simulLen, simulStart, doTraining, doRun));
		} else {
			return false;
		}
	} else {
		return true;
	}
}
void sLogger::Commit() {
	if (dest==LOG_TO_ORCL) {
		OraCommit(db);
	} else {
		fclose(file->file->handle);
	}
}
