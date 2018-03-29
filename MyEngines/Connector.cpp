#include "Connector.h"

sConnector::sConnector(int type_, int fromCore_, int toCore_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Connector.err"))) : dbg_;
	type=type_;

}
sConnector::sConnector(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Connector.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??

	safeCallEB(parms->setKey(parmKey));
	parms->get(&type, "Type");

	switch (type) {
	case CONN_DENSE:
		break;
	case CONN_LINEAR:
		break;
	case CONN_TRANSFORM:
		break;
	default:
		throwE("Invalid Connector Type: %d", 1, type);
		break;
	}
}
sConnector::~sConnector() {}
