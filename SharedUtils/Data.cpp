#include "Data.h"

sData::sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;

}
sData::sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;

	safeCallEB(parms->setKey(parmKey));

	//-- Data Shape
	safeCallEB(parms->setKey("Shape"));
	parms->get(&sampleLen, "SampleLen");
	parms->get(&predictionLen, "PredictionLen");
	parms->get(&featuresCnt, "FeaturesCount");

	//-- Data Actions
	safeCallEE(parms->setKey("..Action"));
	parms->get(&doTrain, "Train");
	parms->get(&doValidation, "Validate");
	parms->get(&doTest, "Test");

	//-- Train Set
	safeCallEE(parms->setKey("..TrainSet"));

	//-- Validation Set
	//-- Test Set

}
sData::~sData() {
	delete dbg;
	/*for (int s=0; s<3; s++) {
		delete ts[s];
		delete ds[s];
	}*/
	delete ts;
	delete ds;
}

void sData::mallocSets() {
	ts=(tTimeSerie**)malloc(3*sizeof(tTimeSerie*));
	ds=(tDataSet**)malloc(3*sizeof(tDataSet*));
}