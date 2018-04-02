#include "DataShape.h"

sDataShape::sDataShape(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DataShape.err"))) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;
}
sDataShape::sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DataShape.err"))) : dbg_;

	safeCallEB(parms->setKey(parmKey));
	safeCallEE(parms->get(&sampleLen, "SampleLen"));
	safeCallEE(parms->get(&predictionLen, "PredictionLen"));
	safeCallEE(parms->get(&featuresCnt, "FeaturesCount"));

}