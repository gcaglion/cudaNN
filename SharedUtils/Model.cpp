#include "Model.h"

sModel::sModel(bool doTrain_, bool doValidation_, bool doTest_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Model.err"))) : dbg_;
	doTrain=doTrain_; doValidation=doValidation_; doTest=doTest_;
}
sModel::sModel(tParmsSource* parms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Model.err"))) : dbg_;

	if (parms->setKey(".Model.Debugger")) {
		parms->newDebugger(dbg);
	} else {
	}
	//-- define model actions
	safeCallEE(parms->setKey(".Model.Action"));
	parms->get(&doTrain, "Train");
	parms->get(&doValidation, "Validate");
	parms->get(&doTest, "Test");

}
sModel::~sModel() {
	delete dbg;
	if(data!=nullptr) delete data;
	if(engine!=nullptr) delete engine;
	if(persistor!=nullptr) delete persistor;
}
