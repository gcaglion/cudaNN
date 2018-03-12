#pragma once
#include "../CommonEnv.h"

#include "Debugger.h"
#include "Data.h"
#include "../MyEngines/Engine.h"
#include "../Logger/Logger.h"

typedef struct sModel {
	tDebugger* dbg;

	tData* data;
	tEngine* engine;
	tLogger* persistor;

	bool doTrain;
	bool doValidation;
	bool doTest;

	EXPORT sModel(bool doTrain_=true, bool doValidation_=false, bool doTest_=true, tDebugger* dbg_=nullptr);
	EXPORT sModel(tParmsSource* parms, tDebugger* dbg_=nullptr);
	EXPORT ~sModel();
} tModel;