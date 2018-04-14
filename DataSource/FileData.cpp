#include "FileData.h"

//=== sFileData
sFileData::sFileData(tFileInfo* srcFile_, int fieldSep_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, tDebugger* dbg_) {
	//--parent DataSource properties
	type=FILE_SOURCE;
	calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
	//--
	srcFile=srcFile_; fieldSep=fieldSep_;
	//-- need to set featuresCnt, but we need to red file to do that!!!
	featuresCnt=-99;
}
sFileData::sFileData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) : sDataSource(dbg_) {
	safeCallEB(parms->setKey(parmKey));
	char ffname[MAX_PATH];
	safeCallEE(srcFile=new tFileInfo(ffname, FILE_MODE_READ));
	parms->get(&fieldSep, "FieldSep");
}
sFileData::~sFileData() {
	delete srcFile;
}
