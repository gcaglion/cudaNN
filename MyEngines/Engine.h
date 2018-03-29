#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "Engine_enums.h"
#include "../SharedUtils/ParamMgr.h"
#include "Core.h"
#include "Connector.h"

#define MAX_CORES_CNT		128
#define MAX_CONNECTORS_CNT	32768

typedef struct sEngine {

	tDebugger* dbg;

	int type;
	int coresCnt;
	int connectorsCnt;

	tCore* core[MAX_CORES_CNT];
	tConnector* connector[MAX_CONNECTORS_CNT];

	EXPORT sEngine(int type_, tDebugger* dbg_);
	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDebugger* dbg_);
	EXPORT ~sEngine();

} tEngine;