#pragma once

#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/ParamMgr.h"
#include "Connector_enums.h"

typedef struct sConnector {

	tDebugger* dbg;

	int type;
	int fromCore;
	int toCore;

	EXPORT sConnector(int type_, int fromCore_, int toCore_, tDebugger* dbg_);
	EXPORT sConnector(tParmsSource* parms, char* parmKey, tDebugger* dbg_);
	EXPORT ~sConnector();

} tConnector;