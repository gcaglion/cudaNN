#include "Core.h"

void sCoreLayout::sCoreLayout_common(tDebugger* dbg_, int Id_) {
	desc=(char*)malloc(CORE_MAX_DESC_LEN);
	parentId=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));
	parentDesc=(char**)malloc(CORE_MAX_PARENTS*sizeof(char*)); for (int p=0; p<CORE_MAX_PARENTS; p++) parentDesc[p]=(char*)malloc(CORE_MAX_DESC_LEN);
	parentConnType=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));

	char fname[MAX_PATH]; sprintf_s(fname, MAX_PATH, "Core%d.err", Id_);
	dbg=(dbg_==nullptr) ? (new tDebugger(fname)) : dbg_;	//... handle specific debugger in xml ...
}

sCoreLayout::sCoreLayout(tParmsSource* parms, int Id_, tDataShape* shape_, tDebugger* dbg_) {
	sCoreLayout_common(dbg_, Id_);
	Id=Id_; shape=shape_;
	sprintf_s(desc, CORE_MAX_DESC_LEN, "Core%d", Id);
	safeCallEB(parms->setKey(desc));

	safeCallEE(parms->get(&type, "Type"));
	safeCallEE(parms->get(parentDesc, "Parents"));
	safeCallEE(parms->get(&parentConnType, "ParentsConnType", &parentsCnt));
	//-- extract parentId from each parentDesc
	for(int p=0; p<parentsCnt; p++){
		parentId[p]=atoi(right(parentDesc[p], (int)strlen(parentDesc[p])-4));
	}

}
sCoreLayout::~sCoreLayout() {
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
sCore::~sCore() {}