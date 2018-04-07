#include "Data.h"


sDataShape::sDataShape(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DataShape.err")) : dbg_;
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;
}
sDataShape::sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DataShape.err")) : dbg_;

	safeCallEB(parms->setKey(parmKey));
	safeCallEE(parms->get(&sampleLen, "SampleLen"));
	safeCallEE(parms->get(&predictionLen, "PredictionLen"));
	safeCallEE(parms->get(&featuresCnt, "FeaturesCount"));

}
sDataShape::~sDataShape() {
	cleanup(dbg);
}

sData::sData(tDataShape* shape_, bool doTrain_, bool doTest_, bool doValidation_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Data.err")) : dbg_;
	shape=shape_; 
	ActionDo[TRAIN]=doTrain_; ActionDo[TEST]=doTest_; ActionDo[VALID]=doValidation_;

}
sData::sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Data.err")) : dbg_;

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
			cleanup(ds[a]);
		}
	}
	cleanup(shape);
	cleanup(dbg);
}
