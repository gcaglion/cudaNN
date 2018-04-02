#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/Generic.h"
#include "../SharedUtils/DataShape.h"
#include "Engine_enums.h"
#include "Layout.h"
#include "Core.h"
#include "../SharedUtils/ParamMgr.h"

typedef struct sEngine {

	tDebugger* dbg;

	int type;

	tDataShape* shape;
	tEngineLayout* layout;

	tCore** core;
	tCoreLayout** coreLayout;

	EXPORT sEngine(int type_, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT ~sEngine();

} tEngine;
