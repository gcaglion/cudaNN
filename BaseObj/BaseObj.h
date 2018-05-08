#pragma once
#include "../CommonEnv.h"
#include "../CommonMacros.h"
#include "../Debugger/Debugger.h"


struct sBaseObj : s0 {


	sDebugger* dbg;
	void spawndbg(sDebuggerParms* dbgparms_);

	EXPORT sBaseObj(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_);

	EXPORT ~sBaseObj();

	void cleanup();

};

