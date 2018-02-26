#include "Logger.h"

sLogger::sLogger(tDBConnection* logDB, tDbg* dbg_) {
	if (dbg_==nullptr) {
		dbg=new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("sLogger.err"));
	} else {
		dbg=dbg_;
	}
	dest=LOG_TO_ORCL;
	db=logDB;
}
sLogger::sLogger(tDataFile* logFile) {
	dest=LOG_TO_TEXT;
	file=logFile;
}
sLogger::~sLogger() { delete dbg; }

void sLogger::SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV) {
	if (saveMSE) {
		if (dest==LOG_TO_ORCL) {
			safeCallEB(Ora_LogSaveMSE(dbg, db, pid, tid, mseCnt, mseT, mseV));
		} else {
		}
	}
}
void sLogger::SaveRun(int pid, int tid, int setid, int npid, int ntid, int runCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual) {
	if (saveRun) {
		if (dest==LOG_TO_ORCL) {
			safeCallEB(Ora_LogSaveRun(dbg, db, pid, tid, setid, npid, ntid, runCnt, featuresCnt, feature, prediction, actual));
		} else {
		}
	}
}

void sLogger::SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W) {
	if (saveImage) {
		numtype* hW;
		#ifdef USE_GPU
			hW=(numtype*)malloc(Wcnt*sizeof(numtype));
			safeCallEB(cudaMemcpy(hW, W, Wcnt*sizeof(numtype), cudaMemcpyDeviceToHost)==cudaSuccess);
		#else
			hW=W;
		#endif
		if (dest==LOG_TO_ORCL) {
			safeCallEB(Ora_LogSaveW(dbg, db, pid, tid, epoch, Wcnt, hW));
		} else {
		}
		
		#ifdef USE_GPU
			free(hW);
		#endif
	} else {
	}
}
void sLogger::LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W) {
	numtype* hW;
#ifdef USE_GPU
	hW=(numtype*)malloc(Wcnt*sizeof(numtype));
#else
	hW=W;
#endif
	if (dest==LOG_TO_ORCL) {
		safeCallEB(Ora_LogLoadW(dbg, db, pid, tid, epoch, Wcnt, hW));
	} else {
	}
#ifdef USE_GPU
	safeCallEB(cudaMemcpy(W, hW, Wcnt*sizeof(numtype), cudaMemcpyHostToDevice)==cudaSuccess);
	free(hW);
#endif
}
void sLogger::SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTrain, bool doTrainRun, bool doTestRun) {
	if (saveClient) {
		if (dest==LOG_TO_ORCL) {
			safeCallEB(Ora_LogSaveClient(dbg, db, pid, clientName, startTime, duration, simulLen, simulStart, doTrain, doTrainRun, doTestRun));
		} else {
		}
	}
}
void sLogger::Commit() {
	if (dest==LOG_TO_ORCL) {
		OraCommit(db);
	} else {
		fclose(file->file->handle);
	}
}
