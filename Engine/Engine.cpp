#include "Engine.h"

//-- Engine stuff
void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_, tDebugger* dbg_) {
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;
	shape=shape_;
}
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_) {
	int c;

	sEngine_common(parms, shape_, dbg_);

	safeCallEB(parms->setKey(parmKey));
	safeCallEB(parms->backupKey());

	safeCallEE(parms->get(&type, "Type"));

	switch (type) {
	case ENGINE_CUSTOM:
		safeCallEB(parms->setKey("Custom"));
		//-- 0. coresCnt
		safeCallEE(parms->get(&coresCnt, "CoresCount"));
		//-- 1. malloc one core and one coreLayout for each core
		core=(tCore**)malloc(coresCnt*sizeof(tCore*));
		coreLayout=(tCoreLayout**)malloc(coresCnt*sizeof(tCoreLayout*));
		//-- 2. create layout, set base coreLayout properties for each Core (type, desc, connType, outputCnt)
		for (c=0; c<coresCnt; c++) {
			safeCallEB(parms->backupKey());
			safeCallEE(coreLayout[c]=new tCoreLayout(parms, c, shape));
			safeCallEB(parms->restoreKey());
		}
		break;
	case ENGINE_WNN:
		safeCallEB(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safeCallEB(parms->setKey("XIE"));
		//... get() ...
		break;
	default:
		throwE("Invalid Engine Type: %d", 1, type);
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
		safeCallEE(parms->backupKey());
		safeCallEE(addCore(parms, c));
		safeCallEE(parms->restoreKey());
	}
}
sEngine::~sEngine() {
	for (int i=0; i<coresCnt; i++) {
		delete coreLayout[i];
//		delete core[i];
	}
	free(core);
	free(coreLayout);
	free(layerCoresCnt);
}


void sEngine::addCore(tParmsSource* parms, int coreId) {

	safeCallEE(parms->setKey(coreLayout[coreId]->desc));

	switch (coreLayout[coreId]->type) {
	case CORE_NN:
		safeCallEE(core[coreId]=new tNN(parms, coreLayout[coreId]));
		break;
	case CORE_GA:
		//safeCallEE(core[coreId]=new tGA(parms, coreLayout[coreId]));
		safeCallEE(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	case CORE_SVM: 
		//safeCallEE(core[coreId]=new tSVM(parms, coreLayout[coreId]));
		safeCallEE(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	case CORE_SOM: 
		//safeCallEE(core[coreId]=new tSOM(parms, coreLayout[coreId]));
		safeCallEE(core[coreId]=new tCore(parms, coreLayout[coreId]));
		break;
	default:
		throwE("coreId %d : invalid coreType: %d", 2, coreLayout[coreId]->Id, coreLayout[coreId]->type);
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
