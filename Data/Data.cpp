#include "Data.h"

#define safemethod(mname, ...) { \
	info("%s->%s() Trying:  %s(%s)...", objName, __func__, #mname, __VA_ARGS__); \
	try { \
		mname(__VA_ARGS__); \
		info("%s->%s() Success: %s(%s)...", objName, __func__, #mname, __VA_ARGS__); \
	} catch (std::exception exc) { \
		fail("%s->%s() Failure: %s(%s) . Exception: ---%s---", objName, __func__, #mname, __VA_ARGS__, exc.what()); \
	} \
}

#define method(mname, ...) { \
	info("%s->%s() Trying:  %s(%s)...", objName, __func__, #mname, __VA_ARGS__); \
	try { \
		mname(__VA_ARGS__); \
	} catch (std::exception exc) { \
		printf("DioPorco!\n"); \
	} \
}



sDataShape::sDataShape(char* objName_, s0* objParent_, int sampleLen_, int predictionLen_, int featuresCnt_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	sampleLen=sampleLen_; predictionLen=predictionLen_; featuresCnt=featuresCnt_;
}
sDataShape::sDataShape(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {

	//safemethod(parms->setKey, parmKey);
	//safemethod(parms->get, &sampleLen, "SampleLen");

	safecall(parms->setKey(parmKey));
	safecall(parms->get(&sampleLen, "SampleLen"));
	safecall(parms->get(&predictionLen, "PredictionLen"));
	safecall(parms->get(&featuresCnt, "FeaturesCount"));

}
sDataShape::~sDataShape() {
}

sData::sData(char* objName_, s0* objParent_, tDataShape* shape_, bool doTrain_, bool doTest_, bool doValidation_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	shape=shape_; 
	ActionDo[TRAIN]=doTrain_; ActionDo[TEST]=doTest_; ActionDo[VALID]=doValidation_;

}
sData::sData(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {

	//-- Shape
	safecall(parms->setKey(parmKey));
	safespawn(shape, tDataShape, parms, "Shape");

	//-- Actions, TimeSeries and DataSets
	for(int a=0; a<3; a++) {
		safecall(parms->setKey(parmKey)); safecall(parms->setKey(ActionDesc[a]));
		safecall(parms->get(&ActionDo[a], "Do"));
		if(ActionDo[a]) {
			safespawn(ds[a], tDataSet, parms, "DataSet");
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
