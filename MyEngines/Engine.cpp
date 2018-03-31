#include "Engine.h"


sEngine::sEngine(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Engine.err"))) : dbg_;
	type=type_; inputCnt=inputCnt_; outputCnt=outputCnt_;

}
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Engine.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??
	
	safeCallEB(parms->setKey(parmKey));
	parms->get(&type, "Type");

	switch (type) {
	case ENGINE_WNN:
		safeCallEB(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safeCallEB(parms->setKey("XIE"));
		coresCnt=3;
		//... get() ...
		break;
	case ENGINE_CUSTOM:
		safeCallEB(parms->setKey("Custom"));
		safeCallEB(parms->backupKey());
		coresCnt=0;

		//-- 1. Connectors
		char connectorKey[XML_MAX_PARAM_NAME_LEN];
		parms->get(&connectorsCnt, "ConnectorsCount"); if (connectorsCnt>MAX_CONNECTORS_CNT) throwE("Engine ConnectorsCount (%d) is larger than maximum allowed (%d)", 2, connectorsCnt, MAX_CONNECTORS_CNT);
		for (int c=0; c<connectorsCnt; c++) {
			sprintf_s(connectorKey, XML_MAX_PARAM_NAME_LEN, "Connector%d", c);
			safeCallEE(connector[c]=new tConnector(parms, connectorKey, dbg));
			//-- list of CoreIds to load is determined by all unique FromCore and ToCore values
			if (!isInList(connector[c]->fromCore, coresCnt, coreId)) {
				coreId[coresCnt]=connector[c]->fromCore;
				coresCnt++;
			}
			if (!isInList(connector[c]->toCore, coresCnt, coreId)) {
				coreId[coresCnt]=connector[c]->toCore;
				coresCnt++;
			}
			safeCallEB(parms->restoreKey());
		}

		// ... TODO ... coresCnt, CoreKeys to load, as well as inputCnt,outputCnt for all Cores should be calcd based on connectors config ....



		//-- 2. Cores
		char coreKey[XML_MAX_PARAM_NAME_LEN];
		parms->get(&coresCnt, "CoresCount"); if (coresCnt>MAX_CORES_CNT) throwE("Engine CoresCount (%d) is larger than maximum allowed (%d)", 2, coresCnt, MAX_CORES_CNT);
		for (int c=0; c<coresCnt; c++) {
			sprintf_s(coreKey, XML_MAX_PARAM_NAME_LEN, "Core%d", c);
			safeCallEE(core[c]=new tCore(parms, coreKey, dbg));
			safeCallEB(parms->restoreKey());
		}

		break;
	default:
		throwE("Invalid Engine Type: %d", 1, type);
		break;
	}
}
sEngine::~sEngine(){}

void sEngine::setLayout(int inputCnt_, int outputCnt_) {
	inputCnt=inputCnt_; outputCnt=outputCnt_;
}
