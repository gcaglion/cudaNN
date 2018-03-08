#include "DataModel.h"

//=== DataModel
sDataModel::sDataModel(int sampleLen_, int predictionLen_, int featuresCnt_, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataModel.err"))) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;
}
sDataModel::sDataModel(tParamMgr* parms, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataModel.err"))) : dbg_;

	parms->sectionSet("Data.Model");
	parms->getx(&sampleLen, "SampleLen");
	parms->getx(&predictionLen, "PredictionLen");
	parms->getx(&featuresCnt, "FeaturesCount");
	parms->getx(&doTrain, "doTrain");
	parms->getx(&doTestOnTrain, "doTestOnTrain");
	parms->getx(&doValid, "doValid");
	parms->getx(&doTest, "doTest");
}
sDataModel::~sDataModel() {
}

