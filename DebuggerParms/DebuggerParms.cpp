#include "DebuggerParms.h"

sDebuggerParms::sDebuggerParms(char* ownerObjName, int dest_, bool verbose_, bool timing_, bool pauseOnError_, char* outFileFullName_, char* outFilePath_) {
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