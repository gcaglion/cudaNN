#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "Core_enums.h"

typedef struct sCore {

	tDebugger* dbg;

	int type;

	int inputCnt;
	int outputCnt;

	EXPORT sCore(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_=nullptr);
	EXPORT sCore(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sCore();

	EXPORT void setLayout(int inputCnt_, int outputCnt_);

} tCore;