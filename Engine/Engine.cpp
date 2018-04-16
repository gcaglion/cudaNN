#include "Engine.h"

//-- Engine stuff
void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_, tDebugger* dbg_) {
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;
	shape=shape_;
}
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_) : sBaseObj("Engine", dbg_) {
	int c;

	sEngine_common(parms, shape_, dbg_);

	safeCall(parms->setKey(parmKey));
	safeCall(parms->backupKey());

	safeCall(parms->get(&type, "Type"));

	switch (type) {
	case ENGINE_CUSTOM:
		safeCall(parms->setKey("Custom"));
		//-- 0. coresCnt
		safeCall(parms->get(&coresCnt, "CoresCount"));
		//-- 1. malloc one core and one coreLayout for each core
		core=(tCore**)malloc(coresCnt*sizeof(tCore*));
		coreLayout=(tCoreLayout**)malloc(coresCnt*sizeof(tCoreLayout*));
		//-- 2. create layout, set base coreLayout properties for each Core (type, desc, connType, outputCnt)
		for (c=0; c<coresCnt; c++) {

			//--
			dbg->write(DBG_LEVEL_STD, "%s -> %s() calling %s ... ", 3, dbg->parentObjName, __func__, "coreLayout[c]=new tCoreLayout(parms, c, shape)");
				if (dbg->timing) dbg->setStartTime();
					try { coreLayout[c]=new tCoreLayout(parms, c, shape); }
			catch (char* exc) {
				printf("\ndbg->parentObjName=%s\n", dbg->parentObjName);
				printf("exc=%s\n", exc);
				sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s -> %s() FAILURE: %s\n", dbg->parentObjName, __func__, exc);
				throw(dbg->errmsg);
			}
			dbg->write(DBG_LEVEL_STD, "SUCCESS.", 0);
			if (dbg->timing) { dbg->setElapsedTime(); dbg->write(DBG_LEVEL_STD, " Elapsed time: %.4f s.", 1, (dbg->elapsedTime/(float)1000)); }
			dbg->write(DBG_LEVEL_STD, "\n", 0);
			//--

			//safeCall(coreLayout[c]=new tCoreLayout(parms, c, shape));
			safeCall(parms->restoreKey());
		}
		break;
	case ENGINE_WNN:
		safeCall(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safeCall(parms->setKey("XIE"));
		//... get() ...
		break;
	default:
		safeThrow("Invalid Engine Type: %d", 1, type);
		break;
	}

	//-- 3. once all coreLayouts are created (and all  parents are set), we can determine Layer for each Core, and cores count for each layer
	for (c=0; c<coresCnt; c++) {
		setCoreLayer(coreLayout[c]);
		layerCoresCnt[coreLayout[c]->layer]++;
	}
	//-- 4. determine layersCnt, and InputCnt for each Core
	for (int l=0; l<MAX_ENGINE_LAYERS; l++) {
		for (c=0; c<layerCoresCnt[l]; c++) {
			if (l==0) {
				//-- do nothing. keep core shape same as engine shape
			} else {
				//-- change sampleLen
				coreLayout[c]->shape->sampleLen=layerCoresCnt[l-1]*coreLayout[c]->shape->predictionLen;
			}
		}
		if (c==0) break;
		layersCnt++;
	}

	//-- 5. add each core
	for (c=0; c<coresCnt; c++) {
		safeCall(parms->backupKey());
		safeCall(addCore(parms, c));
		safeCall(parms->restoreKey());
	}
}

void sEngine::cleanup() {
	for (int i=0; i<coresCnt; i++) {
		delete coreLayout[i];
	}
	free(core);
	free(coreLayout);
	free(layerCoresCnt);
}


void sEngine::addCore(tParmsSource* parms, int coreId) {

	safeCall(parms->setKey(coreLayout[coreId]->desc));

	switch (coreLayout[coreId]->type) {
	case CORE_NN:
		safeCall(core[coreId]=new tNN(parms, coreLayout[coreId]));
		break;
	case CORE_GA:
		//safeCall(core[coreId]=new tGA(parms, coreLayout[coreId]));
		safeCall(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	case CORE_SVM: 
		//safeCall(core[coreId]=new tSVM(parms, coreLayout[coreId]));
		safeCall(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	case CORE_SOM: 
		//safeCall(core[coreId]=new tSOM(parms, coreLayout[coreId]));
		safeCall(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	default:
		safeThrow("coreId %d : invalid coreType: %d", 2, coreLayout[coreId]->Id, coreLayout[coreId]->type);
		break;
	}
}
void sEngine::setCoreLayer(tCoreLayout* c){
	int ret=0;
	int maxParentLayer=-1;
	for (int p=0; p<c->parentsCnt; p++) {
		tCoreLayout* parent=coreLayout[c->parentId[p]];
		setCoreLayer(parent);
		if (parent->layer>maxParentLayer) {
			maxParentLayer=parent->layer;
		}
		ret=maxParentLayer+1;
	}
	c->layer=ret;
}
