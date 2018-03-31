#include "Data.h"

sData::sData(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;

}
sData::sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Data.err"))) : dbg_;

	//-- Shape
	safeCallEB(parms->setKey(parmKey));	safeCallEB(parms->setKey("Shape"));
	safeCallEE(parms->get(&sampleLen, "SampleLen"));
	safeCallEE(parms->get(&predictionLen, "PredictionLen"));
	safeCallEE(parms->get(&featuresCnt, "FeaturesCount"));

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
