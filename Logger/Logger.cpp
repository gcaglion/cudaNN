#include "Logger.h"

sLogger::sLogger(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	safecall(parms->setKey(parmKey));
	parms->get(&saveNothing, "saveNothing");
	parms->get(&saveClient, "saveClient");
	parms->get(&saveMSE, "saveMSE");
	parms->get(&saveRun, "saveRun");
	parms->get(&saveInternals, "saveInternals");
	parms->get(&saveImage, "saveImage");
	parms->get(&dest, "Destination");
	if (dest==ORCL_DEST) {
		safespawn(db, tDBConnection, parms, "DestDB");
	} else {
		safespawn(file, tFileData, parms, "DestFiles");
	}
}
sLogger::sLogger(char* objName_, s0* objParent_, tDBConnection* logDB, Bool saveNothing_, Bool saveClient_, Bool saveMSE_, Bool saveRun_, Bool saveInternals_, Bool saveImage_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	dest=ORCL_DEST;
	db=logDB;
	saveNothing=saveNothing_; saveClient=saveClient_; saveMSE=saveMSE_; saveRun=saveRun_; saveInternals=saveInternals_; saveImage=saveImage_;
}
sLogger::sLogger(char* objName_, s0* objParent_, tFileData* logFile, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	dest=FILE_DEST;
	file=logFile;
}
void sLogger::SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV) {
	if (saveMSE) {
		if (dest==ORCL_DEST) {
			safecall(Ora_LogSaveMSE(Cdbg, db, pid, tid, mseCnt, mseT, mseV));
		} else {
		}
	}
}
void sLogger::SaveRun(int pid, int tid, int setid, int npid, int ntid, int runCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual) {
	if (saveRun) {
		if (dest==ORCL_DEST) {
			safecall(Ora_LogSaveRun(Cdbg, db, pid, tid, setid, npid, ntid, runCnt, featuresCnt, feature, prediction, actual));
		} else {
		}
	}
}

void sLogger::SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W) {
	if (saveImage) {
	
		numtype* hW=(numtype*)malloc(Wcnt*sizeof(numtype));

		if (dest==ORCL_DEST) {
			safecall(Ora_LogSaveW(Cdbg, db, pid, tid, epoch, Wcnt, hW));
		} else {
			fail("%s(%p)->%s() not implemented for FILE_DEST.", name, this, __func__);
		}

		Alg->x2h(hW, W, Wcnt*sizeof(numtype));
		free(hW);
	} 
}
void sLogger::LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W) {

	numtype* hW=(numtype*)malloc(Wcnt*sizeof(numtype));

	if (dest==ORCL_DEST) {
		safecall(Ora_LogLoadW(Cdbg, db, pid, tid, epoch, Wcnt, hW));
	} else {
		fail("%s(%p)->%s() not implemented for FILE_DEST.", name, this, __func__);
	}

	Alg->h2x(W, hW, Wcnt*sizeof(numtype));

	free(hW);
}
void sLogger::SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, Bool doTrain, Bool doTrainRun, Bool doTestRun) {
	if (saveClient) {
		if (dest==ORCL_DEST) {
			safecall(Ora_LogSaveClient(Cdbg, db, pid, clientName, startTime, duration, simulLen, simulStart, doTrain, doTrainRun, doTestRun));
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
