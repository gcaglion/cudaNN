#pragma once

#include "../CommonEnv.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Engine/Engine.h"
#include "../Logger/Logger.h"

typedef struct sForecaster : public sBaseObj {
	
	tData* data=nullptr;
	tEngine* engine=nullptr;
	tLogger* persistor=nullptr;

	EXPORT sForecaster(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey="Forecaster", sDebuggerParms* dbgparms_=nullptr);

} tForecaster;