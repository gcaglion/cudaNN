#include "Data.h"

sData::sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;

	mallocSets();
}
sData::sData(tParmsSource* parms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;

	parms->gotoKey("Data");
	parms->get(&sampleLen, "SampleLen");
	parms->get(&predictionLen, "PredictionLen");
	parms->get(&featuresCnt, "FeaturesCount");

	mallocSets();
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