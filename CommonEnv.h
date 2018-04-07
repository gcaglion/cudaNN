#pragma once

#include "CommonEnv00.h"
#include "Debugger/Debugger.h"

#ifdef __cplusplus
#include <stdexcept>

class sBaseObj {
public:
	tDebugger* dbg=nullptr;
	virtual ~sBaseObj() {}
};
#endif
