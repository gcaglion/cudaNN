#include "Core.h"

void sCoreLayout::sCoreLayout_common(tDebugger* dbg_, int Id_) {
	desc=(char*)malloc(CORE_MAX_DESC_LEN);
	parentId=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));
	parentDesc=(char**)malloc(CORE_MAX_PARENTS*sizeof(char*)); for (int p=0; p<CORE_MAX_PARENTS; p++) parentDesc[p]=(char*)malloc(CORE_MAX_DESC_LEN);
	parentConnType=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));

	char fname[MAX_PATH]; sprintf_s(fname, MAX_PATH, "Core%d.err", Id_);
}

sCoreLayout::sCoreLayout(tParmsSource* parms, int Id_, tDataShape* shape_, tDebugger* dbg_) : sBaseObj("CoreLayout", dbg_) {
	sCoreLayout_common(dbg_, Id_);
	Id=Id_; shape=shape_;
	sprintf_s(desc, CORE_MAX_DESC_LEN, "Core%d", Id);

	//safeCall(parms->setKey(desc));
	try {
		parms->setKey(desc);
	}
	catch (char* exc) {
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s -> %s() FAILURE: %s\n", dbg->parentObjName, __func__, exc);
		throw dbg->errmsg;
	}

	safeCall(parms->get(&type, "Type"));
	safeCall(parms->get(parentDesc, "Parents"));
	safeCall(parms->get(&parentConnType, "ParentsConnType", &parentsCnt));
	//-- extract parentId from each parentDesc
	for(int p=0; p<parentsCnt; p++){
		parentId[p]=atoi(right(parentDesc[p], (int)strlen(parentDesc[p])-4));
	}

}
sCoreLayout::~sCoreLayout() {
	cleanup();
}
void sCoreLayout::cleanup() {
	free(desc);
	free(parentId);
	free(parentConnType);
	for (int p=0; p<CORE_MAX_PARENTS; p++) free(parentDesc[p]);
	free(parentDesc);
}

sCore::sCore() {}
sCore::sCore(tParmsSource* parms, tCoreLayout* layout_) {
	layout=layout_;
}
