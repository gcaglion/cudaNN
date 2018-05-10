#include "DebuggerParms.h"

sDebuggerParms::sDebuggerParms(char* ownerObjName, int dest_, Bool verbose_, Bool timing_, Bool pauseOnError_, char* outFileFullName_, char* outFilePath_) {
	dest=dest_;
	verbose=verbose_;
	timing=timing_;
	pauseOnError=pauseOnError_;
	if (ownerObjName==nullptr) {
		strcpy_s(outFileName, MAX_PATH, DEFAULT_DBG_FNAME);
	} else {
		sprintf_s(outFileName, MAX_PATH, "%s_Debugger", ownerObjName);
	}
	if (outFilePath_==nullptr) {
		strcpy_s(outFilePath, MAX_PATH, DEFAULT_DBG_FPATH);
	} else {
		strcpy_s(outFilePath, MAX_PATH, outFilePath_);
	}
	if (outFileFullName_==nullptr) {
		sprintf_s(outFileFullName, MAX_PATH, "%s/%s(%p).%s", outFilePath, outFileName, this, (verbose) ? "log" : "err");
	} else {
		strcpy_s(outFileFullName, MAX_PATH, outFileFullName_);
	}
	outFile=nullptr;
}
sDebuggerParms::~sDebuggerParms() {
	delete outFile;
}

//-- C stuff
/*
EXPORT void _foutC(tDebuggerC* dbg, Bool success) {
	for (int t=0; t<dbg->stackLevel; t++) sprintf_s(dbg->dbgmsg, DBG_MSG_MAXLEN, "\t%s", dbg->dbgmsg);
	strcat_s(dbg->dbgmsg, DBG_MSG_MAXLEN, "\n");
	strcat_s(dbg->stackmsg, DBG_STACK_MAXLEN, dbg->dbgmsg);
	//if (stackLevel>0) sprintf_s(parent->stackmsg, DBG_STACK_MAXLEN, "%st%s", dbg->parent->stackmsg, dbg->dbgmsg);
	printf("%s", dbg->dbgmsg);
	if (dbg->dbgparms->outFile!=NULL) fprintf(dbg->dbgparms->outFile->handle, "%s", dbg->dbgmsg);
	if (!success && dbg->dbgparms->pauseOnError) { printf("Press any key..."); getchar(); }
}
*/