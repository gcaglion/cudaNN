#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/Generic.h"
#include "../SharedUtils/DataShape.h"
#include "Engine_enums.h"
#include "EngineLayout.h"
#include "../SharedUtils/ParamMgr.h"
#include "Core.h"
#include "Connector.h"

#define MAX_CORES_CNT		128
#define MAX_CONNECTORS_CNT	MAX_CORES_CNT*MAX_CORES_CNT

typedef struct sEngine {

	tDebugger* dbg;

	int type;

	tDataShape* shape;
	tEngineLayout* layout;
	tCore** core;

	EXPORT sEngine(int type_, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT ~sEngine();

} tEngine;
