#include "Model.h"

sModel::sModel(bool doTrain_, bool doValidation_, bool doTest_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("Model.err"))) : dbg_;
	doTrain=doTrain_; doValidation=doValidation_; doTest=doTest_;
}
sModel::sModel(tParmsSource* parms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("Model.err"))) : dbg_;

	safeCallEB(parms->gotoKey("Model", true, false));
	
	if (parms->gotoKey("Debugger", false, true)) {
		parms->newDebugger(dbg);
	} else {
	}
	parms->get(&doTrain, "Model", "doTrain");
	parms->get(&doValidation, "Model", "doValidation");
	parms->get(&doTest, "Model", "doTest");

}
sModel::~sModel() {
	delete dbg;
	if(data!=nullptr) delete data;
	if(engine!=nullptr) delete engine;
	if(persistor!=nullptr) delete persistor;
}
