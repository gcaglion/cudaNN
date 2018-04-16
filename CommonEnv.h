#pragma once

#include "CommonEnv00.h"
#include "Debugger/Debugger.h"

#ifdef __cplusplus

class sBaseObj {
public:
	char objName[MAX_PATH];
	tDebugger* dbg;

	sBaseObj(char* className, tDebugger* dbg_) {
		sprintf_s(objName, MAX_PATH, "%s_%p", className, this);
		if (dbg_==nullptr) {
			dbg=new tDebugger(DBG_DEFAULT_LEVEL, DBG_DEFAULT_DEST, objName);
		} else {
			dbg=dbg_;
		}
	}

	~sBaseObj() {
		delete dbg;
	}

	virtual void cleanup() {}

};
#endif
