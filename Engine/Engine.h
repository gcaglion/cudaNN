#pragma once
#include "../CommonEnv.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Core/Core.h"

#include "../cuNN/cuNN.h"

#define MAX_ENGINE_CORES	32

typedef struct sEngine {

	int coresCnt;

	tCore** core;


	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_);
	EXPORT ~sEngine();
	EXPORT void addCore(int coreType);

} tEngine;