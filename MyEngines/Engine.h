#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/Generic.h"
#include "Engine_enums.h"
#include "../SharedUtils/ParamMgr.h"
#include "Core.h"
#include "Connector.h"

#define MAX_CORES_CNT		128
#define MAX_CONNECTORS_CNT	32768

typedef struct sEngine {

	tDebugger* dbg;

	int type;
	int inputCnt;
	int outputCnt;

	int coresCnt;
	int coreId[MAX_CORES_CNT];
	tCore* core[MAX_CORES_CNT];
	int connectorsCnt;
	tConnector* connector[MAX_CONNECTORS_CNT];

	EXPORT sEngine(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_=nullptr);
	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sEngine();

	EXPORT void setLayout(int inputCnt_, int outputCnt_);

} tEngine;