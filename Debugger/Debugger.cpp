#include "Debugger.h"

sDebuggerParms::sDebuggerParms(bool verbose_, bool timing_, bool pauseOnError_) {
	verbose=verbose_;
	timing=timing_;
	pauseOnError=pauseOnError_;
}

sDebugger::sDebugger(char* outFileName, sDebuggerParms* parms_, char* outFilePath) {
	if (parms_==nullptr) {
		parms=new sDebuggerParms();
	} else {
		parms=parms_;
	}
	char outfname[MAX_PATH];
	sprintf_s(outfname, MAX_PATH, "%s/%s(%p).%s", outFilePath, outFileName, this, (parms->verbose) ? "log" : "err");

	try {
		spawnFile(outFile, outfname, FILE_MODE_WRITE);
	}
	catch (std::exception exc) {
		err_d("sDebugger(%p)->%s() failed. Error creating debugger outFile %s ...\n", this, __func__, outfname);
		throw(exc);
	}
}
sDebugger::~sDebugger() {
	delete outFile;
}
