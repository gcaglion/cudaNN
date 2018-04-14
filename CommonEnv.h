#pragma once

#include "CommonEnv00.h"
#include "Debugger/Debugger.h"

#ifdef __cplusplus
#include <stdexcept>

class sBaseObj {
public:
	char objName[MAX_PATH];
	tDebugger* dbg;

	sBaseObj(char* className, tDebugger* dbg_) {
		if (dbg_==nullptr) {
			sprintf_s(objName, MAX_PATH, "%s_%p.log", className, this);
			dbg=new tDebugger(DBG_LEVEL_DEFAULT, DBG_DEST_DEFAULT, objName);
		} else {
			dbg=dbg_;
		}
	}

	~sBaseObj() {
		delete dbg;
	}


};
#endif
