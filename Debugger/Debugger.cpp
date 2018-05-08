#include "Debugger.h"


sDebugger::sDebugger(char* objName_, s0* objParent_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {

	safecall(spawnFile(outFile, parms->outFileFullName, FILE_MODE_WRITE));

}

sDebugger::~sDebugger() {
	delete outFile;
}
