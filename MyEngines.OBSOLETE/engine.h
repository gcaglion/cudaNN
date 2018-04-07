#pragma once
#include "../CommonEnv.h"
#include "../Debugger/Debugger.h"
#include "../Utils/Utils.h"
#include "../Data/Data.h"
#include "Engine_enums.h"
#include "Layout.h"
#include "Core.h"
#include "../ParamMgr/ParamMgr.h"

#include "../cuNN/cuNN.h"

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

	EXPORT void train(tDataSet* trainDS);
	EXPORT void addCore(tParmsSource* parms, int coreId, int coreType);

} tEngine;
