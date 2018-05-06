#include "Engine.h"

//-- Engine stuff
void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_) {
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;
	shape=shape_;
}
sEngine::sEngine(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, tDataShape* shape_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	int c;

	//-- look for object-specific debugger properties in parms file. If they are there, then create a new debugger and replace the default one
	parms->ObjectDebugParmsOverride(parmKey, dbg);

	sEngine_common(parms, shape_);

	safecall(parms->setKey(parmKey));

	safecall(parms->get(&type, "Type"));

	switch (type) {
	case ENGINE_CUSTOM:
		safecall(parms->setKey("Custom"));
		safecall(parms->backupKey());
		//-- 0. coresCnt
		safecall(parms->get(&coresCnt, "CoresCount"));
		//-- 1. malloc one core and one coreLayout for each core
		core=(tCore**)malloc(coresCnt*sizeof(tCore*));
		coreLayout=(tCoreLayout**)malloc(coresCnt*sizeof(tCoreLayout*));
		//-- 2. create layout, set base coreLayout properties for each Core (type, desc, connType, outputCnt)
		for (c=0; c<coresCnt; c++) {
			safespawn(coreLayout[c], sCoreLayout, parms, c, shape);
			safecall(parms->restoreKey());
		}
		break;
	case ENGINE_WNN:
		safecall(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safecall(parms->setKey("XIE"));
		//... get() ...
		break;
	default:
		fail("Invalid Engine Type: %d", type);
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
		safecall(parms->backupKey());
		safecall(addCore(parms, c));
		safecall(parms->restoreKey());
	}
}

void sEngine::cleanup() {
	free(core);
	free(coreLayout);
	free(layerCoresCnt);
}


void sEngine::addCore(tParmsSource* parms, int coreId) {

	char cdesc[CORE_MAX_DESC_LEN];
	safecall(parms->setKey(coreLayout[coreId]->desc));

	switch (coreLayout[coreId]->type) {
	case CORE_NN:
		sprintf_s(cdesc, CORE_MAX_DESC_LEN, "Core%d(NN)", coreId);
		safespawn(core[coreId], tNN, parms, coreLayout[coreId]);
		break;
	case CORE_GA:
		sprintf_s(cdesc, CORE_MAX_DESC_LEN, "Core%d(GA)", coreId);
		//safespawn(core[coreId], tGA, parms, coreLayout[coreId]);
		break;
	case CORE_SVM: 
		break;
	case CORE_SOM: 
		break;
	default:
		fail("coreId %d : invalid coreType: %d", coreLayout[coreId]->Id, coreLayout[coreId]->type);
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
