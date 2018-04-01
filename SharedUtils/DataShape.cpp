#include "DataShape.h"

sDataShape::sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DataShape.err"))) : dbg_;

	safeCallEB(parms->setKey(parmKey));
	safeCallEE(parms->get(&sampleLen, "SampleLen"));
	safeCallEE(parms->get(&predictionLen, "PredictionLen"));
	safeCallEE(parms->get(&featuresCnt, "FeaturesCount"));

}