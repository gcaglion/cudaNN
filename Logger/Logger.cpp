#include "Logger.h"

sLogger::sLogger(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("sLogger.err")) : dbg_;
	safeCallEB(parms->setKey(parmKey));
	parms->get(&saveNothing, "saveNothing");
	parms->get(&saveClient, "saveClient");
	parms->get(&saveMSE, "saveMSE");
	parms->get(&saveRun, "saveRun");
	parms->get(&saveInternals, "saveInternals");
	parms->get(&saveImage, "saveImage");
	parms->get(&dest, "Destination");
	if (dest==ORCL_DEST) {
		safeCallEE(db=new tDBConnection(parms, "DestDB"));
	} else {
		safeCallEE(file=new tFileData(parms, "DestFiles"));
	}
}
sLogger::sLogger(tDBConnection* logDB, bool saveNothing_, bool saveClient_, bool saveMSE_, bool saveRun_, bool saveInternals_, bool saveImage_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("sLogger.err")) : dbg_;
	dest=ORCL_DEST;
	db=logDB;
	saveNothing=saveNothing_; saveClient=saveClient_; saveMSE=saveMSE_; saveRun=saveRun_; saveInternals=saveInternals_; saveImage=saveImage_;
}
sLogger::sLogger(tFileData* logFile, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("sLogger.err")) : dbg_;
	dest=FILE_DEST;
	file=logFile;
}
sLogger::~sLogger() {
	cleanup(file);
	cleanup(db);
	cleanup(dbg);
}
void sLogger::SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV) {
	if (saveMSE) {
		if (dest==ORCL_DEST) {
			safeCallEB(Ora_LogSaveMSE(dbg, db, pid, tid, mseCnt, mseT, mseV));
		} else {
		}
	}
}
void sLogger::SaveRun(int pid, int tid, int setid, int npid, int ntid, int runCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual) {
	if (saveRun) {
		if (dest==ORCL_DEST) {
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
		if (dest==ORCL_DEST) {
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
	if (dest==ORCL_DEST) {
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
		if (dest==ORCL_DEST) {
			safeCallEB(Ora_LogSaveClient(dbg, db, pid, clientName, startTime, duration, simulLen, simulStart, doTrain, doTrainRun, doTestRun));
		} else {
		}
	}
}
void sLogger::Commit() {
	if (dest==ORCL_DEST) {
		OraCommit(db);
	} else {
		fclose(file->srcFile->handle);
	}
}
