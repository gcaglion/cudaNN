#include "Data.h"


sDataShape::sDataShape(int sampleLen_, int predictionLen_, int featuresCnt_, tDebugger* dbg_) : sBaseObj("DataShape", dbg_) {
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;
}
sDataShape::sDataShape(tParmsSource* parms, char* parmKey, tDebugger* dbg_) : sBaseObj("DataShape", dbg_) {

	safeCall(parms->setKey(parmKey));
	safeCall(parms->get(&sampleLen, "SampleLen"));
	safeCall(parms->get(&predictionLen, "PredictionLen"));
	safeCall(parms->get(&featuresCnt, "FeaturesCount"));

}
sDataShape::~sDataShape() {
}

sData::sData(tDataShape* shape_, bool doTrain_, bool doTest_, bool doValidation_, tDebugger* dbg_) : sBaseObj("Data", dbg_) {
	shape=shape_; 
	ActionDo[TRAIN]=doTrain_; ActionDo[TEST]=doTest_; ActionDo[VALID]=doValidation_;

}
sData::sData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) : sBaseObj("Data", dbg_) {

	//-- Shape
	safeCall(parms->setKey(parmKey));
	safeCall(shape=new tDataShape(parms, "Shape"));

	//-- Actions, TimeSeries and DataSets
	for(int a=0; a<3; a++) {
		safeCall(parms->setKey(parmKey)); safeCall(parms->setKey(ActionDesc[a]));
		safeCall(parms->get(&ActionDo[a], "Do"));
		if(ActionDo[a]) {
			safeCall(ds[a]=new tDataSet(parms, "DataSet"));
		}
	}

}
sData::~sData() {
	for (int a=0; a<3; a++) {
		if (ActionDo[a]) {
			delete ds[a];
		}
	}
	delete shape;
}
