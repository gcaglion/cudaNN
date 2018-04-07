#include "Layout.h"

sEngineLayout::sEngineLayout(int coresCnt_) {
	coresCnt=coresCnt_;
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;

	coreLayer=(int*)malloc(coresCnt*sizeof(int));
	coreType=(int*)malloc(coresCnt*sizeof(int));
	coreDesc=(char**)malloc(coresCnt*sizeof(char*));
	coreParentsCnt=(int*)malloc(coresCnt*sizeof(int));
	coreParent=(int**)malloc(coresCnt*sizeof(int*));
	coreParentConnType=(int**)malloc(coresCnt*sizeof(int*));
	coreParentDesc=(char***)malloc(coresCnt*sizeof(char**));
	coreInputCnt=(int*)malloc(coresCnt*sizeof(int));
	coreOutputCnt=(int*)malloc(coresCnt*sizeof(int));
	for (int c=0; c<coresCnt; c++) {
		coreDesc[c]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
		coreParent[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentConnType[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentDesc[c]=(char**)malloc(coresCnt*sizeof(char*));
		for (int cc=0; cc<coresCnt; cc++) coreParentDesc[c][cc]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
	}
}
sEngineLayout::~sEngineLayout() {
	for (int c=0; c<coresCnt; c++) {
		free(coreDesc[c]);
		free(coreParent[c]);
		free(coreParentConnType[c]);
		for (int cc=0; cc<coresCnt; cc++) free(coreParentDesc[c][cc]);
		free(coreParentDesc[c]);
	}
	free(coreLayer);
	free(coreType);
	free(coreDesc);
	free(coreParent);
	free(coreParentsCnt);
	free(coreParentConnType);
	free(coreParentDesc);
	free(coreInputCnt);
	free(coreOutputCnt);
	free(layerCoresCnt);
}
int sEngineLayout::getCoreLayer(int c) {
	int ret=0;
	if (coreParentsCnt[c]==0) {
		ret=0;
	} else {
		for (int p=0; p<coreParentsCnt[c]; p++) {
			ret++;
			//coreParent[p]
		}
	}
	return ret;
}

sCoreLayout::sCoreLayout(int id_, char* desc_, int layer_, int type_, int inputCnt_, int outputCnt_, int parentsCnt_, int* parentId_, int* parentConnType_) {
	id=id_; desc=desc_; layer=layer_; type=type_, parentsCnt=parentsCnt_;
	parentId=parentId_;
	parentConnType=parentConnType_;
	inputCnt=inputCnt_; outputCnt=outputCnt_;
}
sCoreLayout::sCoreLayout(tEngineLayout* engineLayout, int coreId) {
	id=coreId;
	desc=engineLayout->coreDesc[id];
	layer= engineLayout->coreLayer[id];
	type= engineLayout->coreType[id];
	parentsCnt=engineLayout->coreParentsCnt[id];
	parentId=engineLayout->coreParent[id];
	parentConnType=engineLayout->coreParentConnType[id];
}
sCoreLayout::~sCoreLayout() {
}
