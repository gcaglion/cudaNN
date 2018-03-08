#include "FileData.h"

//=== sFileData
sFileData::sFileData(tFileInfo* srcFile_, int fieldSep_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, tDbg* dbg_) {
	//--parent DataSource properties
	type=SOURCE_DATA_FROM_FILE;
	calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
	//--
	srcFile=srcFile_; fieldSep=fieldSep_;
	//-- need to set featuresCnt, but we need to red file to do that!!!
	featuresCnt=-99;
}
sFileData::sFileData(tParamMgr* parms, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("FileData.err"))) : dbg_;
	parms->sectionSetChild("FileData");
	char ffname[MAX_PATH];
	safeCallEE(srcFile=new tFileInfo(ffname, FILE_MODE_READ));
	parms->getx(&fieldSep, "FieldSep", true);
}

