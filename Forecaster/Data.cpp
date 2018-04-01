#include "Data.h"

sData::sData(tDataShape* shape_, bool doTrain_, bool doTest_, bool doValidation_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;
	shape=shape_; 
	ActionDo[TRAIN]=doTrain_; ActionDo[TEST]=doTest_; ActionDo[VALID]=doValidation_;

}
sData::sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;

	//-- Shape
	safeCallEB(parms->setKey(parmKey));
	safeCallEE(shape=new tDataShape(parms, "Shape"));

	//-- Actions, TimeSeries and DataSets
	for(int a=0; a<3; a++) {
		safeCallEB(parms->setKey(parmKey)); safeCallEE(parms->setKey(ActionDesc[a]));
		safeCallEE(parms->get(&ActionDo[a], "Do"));
		if(ActionDo[a]) {
			safeCallEE(ds[a]=new tDataSet(parms, "DataSet"));
		}
	}

}
sData::~sData() {
	for (int a=0; a<3; a++) {
		if (ActionDo[a]) {
			delete ds[a];
		}
	}

	delete dbg;
}
