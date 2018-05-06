#include "Debugger.h"


sDebugger::sDebugger(sDebuggerParms* dbgparms_) {
	
	if (dbgparms_==nullptr) {
		parms=new sDebuggerParms();
	} else {
		parms=dbgparms_;
	}

	try {
		spawnFile(outFile, parms->outFileFullName, FILE_MODE_WRITE);
	}
	catch (std::exception exc) {
		err_d("sDebugger(%p)->%s() failed. Error creating debugger outFile %s ...\n", this, __func__, parms->outFileFullName);
		throw(exc);
	}
}

sDebugger::~sDebugger() {
	delete outFile;
}
