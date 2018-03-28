#include "Model.h"

sModel::sModel(bool doTrain_, bool doValidation_, bool doTest_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Model.err"))) : dbg_;
	doTrain=doTrain_; doValidation=doValidation_; doTest=doTest_;
}
sModel::sModel(tParmsSource* parms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Model.err"))) : dbg_;

	safeCallEE(parms->setKey("Model", true, false));
	
	if (parms->setKey("Debugger", false, true)) {
		parms->newDebugger(dbg);
	} else {
	}
	//-- define model actions
	safeCallEE(parms->setKey("Model.Action", true, false));
	parms->get(&doTrain, "Train", true);
	parms->get(&doValidation, "Validation", true);
	parms->get(&doTest, "Test", true);

}
sModel::~sModel() {
	delete dbg;
	if(data!=nullptr) delete data;
	if(engine!=nullptr) delete engine;
	if(persistor!=nullptr) delete persistor;
}
