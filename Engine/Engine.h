#pragma once
#include "../CommonEnv.h"
#include "../DebuggerParms/DebuggerParms.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Core/Core.h"
#include "Engine_enums.h"

#include "../cuNN/cuNN.h"

#define MAX_ENGINE_LAYERS	8
#define MAX_ENGINE_CORES	32

typedef struct sEngine : public s0 {

	int type;
	int coresCnt;
	int layersCnt=0;
	int* layerCoresCnt;

	tDataShape* shape;

	tCoreLayout** coreLayout;
	tCore** core;

	EXPORT void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_);
	EXPORT sEngine(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, tDataShape* shape_, sDebuggerParms* dbgparms_=nullptr);

	EXPORT void setCoreLayer(tCoreLayout* c);
	EXPORT void train(tDataSet* trainDS);
	EXPORT void addCore(tParmsSource* parms, int coreId);

	void cleanup();

} tEngine;