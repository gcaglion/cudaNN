#pragma once

#include "../CommonEnv.h"
#include "../Data/Data.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Logger/Logger.h"
#include "../Engine/Engine.h"

typedef struct sForecaster : public sBaseObj {
	
	tData* data=nullptr;
	tEngine* engine=nullptr;
	tLogger* persistor=nullptr;

	EXPORT sForecaster(tDebugger* dbg_=nullptr);
	EXPORT sForecaster(tParmsSource* parms, char* parmKey="Forecaster", tDebugger* dbg_=nullptr);
	EXPORT ~sForecaster();

} tForecaster;