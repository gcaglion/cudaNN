#pragma once

#include "../CommonEnv.h"
#include "../Data/Data.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Logger/Logger.h"
#include "../Engine/Engine.h"

typedef struct sForecaster {
	
	tDebugger* dbg;

	tData* data;
	tEngine* engine;
	tLogger* persistor;

	EXPORT sForecaster(tDebugger* dbg_=nullptr);
	EXPORT sForecaster(tParmsSource* parms, char* parmKey="Forecaster", tDebugger* dbg_=nullptr);
	EXPORT ~sForecaster();

} tForecaster;