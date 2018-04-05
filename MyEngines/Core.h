#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "Core_enums.h"
#include "Layout.h"
#include "../TimeSerie/DataSet.h"

typedef struct sCore {

	tDebugger* dbg;

	tCoreLayout* layout;

	EXPORT sCore(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_=nullptr);
	EXPORT sCore(tParmsSource* parms, tCoreLayout* layout_, tDebugger* dbg_=nullptr);
	EXPORT sCore();
	EXPORT ~sCore();

} tCore;