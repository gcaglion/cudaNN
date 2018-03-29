#include "Engine.h"


sEngine::sEngine(int type_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Engine.err"))) : dbg_;
	type=type_;

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

		//-- 1. Cores
		char coreKey[XML_MAX_PARAM_NAME_LEN];
		parms->get(&coresCnt, "CoresCount"); if (coresCnt>MAX_CORES_CNT) throwE("Engine CoresCount (%d) is larger than maximum allowed (%d)", 2, coresCnt, MAX_CORES_CNT);
		for (int c=0; c<coresCnt; c++) {
			sprintf_s(coreKey, XML_MAX_PARAM_NAME_LEN, "%d", c);
			core[c]=new tCore(parms, coreKey, dbg);
		}
		
		//-- 2. Connectors
		char connectorKey[XML_MAX_PARAM_NAME_LEN];
		parms->get(&connectorsCnt, "ConnectorsCount"); if (connectorsCnt>MAX_CONNECTORS_CNT) throwE("Engine ConnectorsCount (%d) is larger than maximum allowed (%d)", 2, connectorsCnt, MAX_CONNECTORS_CNT);
		for (int c=0; c<connectorsCnt; c++) {
			sprintf_s(connectorKey, XML_MAX_PARAM_NAME_LEN, "%d", c);
			connector[c]=new tConnector(parms, connectorKey, dbg);
		}

		break;
	default:
		throwE("Invalid Engine Type: %d", 1, type);
		break;
	}
}
sEngine::~sEngine(){}
