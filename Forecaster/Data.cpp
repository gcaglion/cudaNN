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

	//-- TimeSerie(s)
	if (doTrain)		safeCallEE(trainTS=new tTimeSerie(parms, "..TrainSet"));
	if (doTest)			safeCallEE(testTS =new tTimeSerie(parms, "..TestSet"));
	if (doValidation)	safeCallEE(validTS=new tTimeSerie(parms, "..ValidationSet"));

}
sData::~sData() {
	delete dbg;
	if (doTrain) delete trainTS;
	if (doTest) delete testTS;
	if (doValidation) delete validTS;
}
