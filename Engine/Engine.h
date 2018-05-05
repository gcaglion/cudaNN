#pragma once
#include "../CommonEnv.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Core/Core.h"
#include "Engine_enums.h"

#include "../cuNN/cuNN.h"

#define MAX_ENGINE_LAYERS	8
#define MAX_ENGINE_CORES	32

typedef struct sEngine : public sBaseObj {

	int type;
	int coresCnt;
	int layersCnt=0;
	int* layerCoresCnt;

	tDataShape* shape;

	tCoreLayout** coreLayout;
	tCore** core;

	EXPORT void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_);
	EXPORT sEngine(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, tDataShape* shape_, sDebuggerParms* dbgparms_=nullptr);

	EXPORT void setCoreLayer(tCoreLayout* c);
	EXPORT void train(tDataSet* trainDS);
	EXPORT void addCore(tParmsSource* parms, int coreId);

	void cleanup();

} tEngine;