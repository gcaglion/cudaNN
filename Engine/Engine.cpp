#include "Engine.h"

sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_) {
	core=(tCore**)malloc(MAX_ENGINE_CORES*sizeof(tCore*));
	coresCnt=0;
}
sEngine::~sEngine() {
	for (int i=0; i<coresCnt; i++) delete core[i];
	delete core;
}

void sEngine::addCore(int coreType) {
	switch (coreType) {
	case 0:
		//core[coresCnt]=new tNN(200,50,4,nullptr);
		coresCnt++;
		break;
	default:
		break;
	}
	coresCnt++;
}
