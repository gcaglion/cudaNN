#pragma once

#include "CommonEnv00.h"
#include "Debugger/Debugger.h"

#ifdef __cplusplus
#include <stdexcept>

class sBaseObj {
public:
	tDebugger* dbg;

	sBaseObj() {
		char fname[MAX_PATH];
		sprintf_s(fname, MAX_PATH, "Obj_%p.log", this);
		dbg=new tDebugger(DBG_LEVEL_DEFAULT, DBG_DEST_DEFAULT, fname);
	}
	virtual ~sBaseObj() {
		delete dbg;
	}
};
#endif
