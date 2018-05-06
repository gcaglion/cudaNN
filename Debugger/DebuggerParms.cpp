#include "DebuggerParms.h"

sDebuggerParms::sDebuggerParms(int dest_, bool verbose_, bool timing_, bool pauseOnError_, char* outFileFullName_, char* outFilePath_, char* outFileName_) {
	dest=dest_;
	verbose=verbose_;
	timing=timing_;
	pauseOnError=pauseOnError_;
	if (outFileName_==nullptr) {
		strcpy_s(outFileName, MAX_PATH, DEFAULT_DBG_FNAME);
	} else {
		strcpy_s(outFileName, MAX_PATH, outFileName_);
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

}

