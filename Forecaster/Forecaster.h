#pragma once

#include "../CommonEnv.h"
#include "../SharedUtils/Debugger.h"
#include "../SharedUtils/Data.h"
#include "../MyEngines/Engine.h"
#include "../Logger/Logger.h"

typedef struct sForecaster {
	
	tDebugger* dbg;

	tData* data;
	tEngine* engine;
	tLogger* persistor;

	EXPORT sForecaster(tDebugger* dbg_=nullptr);
	EXPORT sForecaster(tParmsSource* parms, char* parmKey="Forecaster", tDebugger* dbg_=nullptr);
	EXPORT ~sForecaster();

} tForecaster;