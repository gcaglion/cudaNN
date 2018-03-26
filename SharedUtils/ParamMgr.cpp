#include "ParamMgr.h"
#include "MyEnums.h"

sParmsSource::sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("ParmsSource.err"))) : dbg_;
	CLoverridesCnt=CLoverridesCnt_; CLoverride=CLoverride_;
	safeCallEE(parmsFile=new tFileInfo(pFileFullName, FILE_MODE_READ));

	currentKey=new tKey();
	soughtKey=new tKey();
	tmpKey=new tKey();
	currentParm=new tParm();
	soughtParm=new tParm();
	tmpParm=new tParm();

	pArrDesc=(char**)malloc(ARRAY_PARAMETER_MAX_ELEMS*sizeof(char*)); for (int i=0; i<ARRAY_PARAMETER_MAX_ELEMS; i++) pArrDesc[i]=(char*)malloc(MAX_PARAMDESC_LEN);

}
void sParmsSource::newDebugger(tDebugger* dbg_) {
	int dbg_level, dbg_dest; char dbg_fname[MAX_PATH];

	get(&dbg_level,"Level", false, enumlist);
	get(&dbg_dest, "Dest", false, enumlist);
	get(dbg_fname, "DestFileFullName");

	safeCallEE(dbg_=new tDebugger(dbg_level, dbg_dest, new tFileInfo(dbg_fname, FILE_MODE_WRITE)));

}
bool sParmsSource::gotoKey(char* soughtKeyDesc, bool fromRoot, bool ignoreError) {

	soughtKey->setFromDesc(soughtKeyDesc, fromRoot);

	if (fromRoot) rewind(parmsFile->handle);

	//-- go to current key start
	if (soughtKey->find(parmsFile)) {
		//-- if found, set it as current
		if (fromRoot) {
			soughtKey->copyTo(currentKey);
		} else {
			soughtKey->appendTo(currentKey);
		}
	} else {
		throwE("could not find key", 0);
	}
	return true;
}

//-- specific, single value: int(with or without enums), numtype, char*
void sParmsSource::get_(char* pvalS, int* oparamVal, bool isenum, int* oListLen) {
	if (isenum) {
		safeCallEB(decode(soughtParm->val, pvalS, oListLen, oparamVal));
	} else {
		(*oparamVal)=atoi(pvalS);
	}
}
void sParmsSource::get_(char* pvalS, bool* oparamVal, bool isenum, int* oListLen) {
	Trim(pvalS); UpperCase(pvalS);
	(*oparamVal)=(strcmp(pvalS, "TRUE")==0);
}
void sParmsSource::get_(char* pvalS, char* oparamVal, bool isenum, int* oListLen) {
	strcpy_s(oparamVal, XML_MAX_PARAM_VAL_LEN, pvalS);
}
void sParmsSource::get_(char* pvalS, numtype* oparamVal, bool isenum, int* oListLen) {
	(*oparamVal)=(numtype)atof(pvalS);
}

//-- specific, arrays: int(with or without enums), numtype, char*
void sParmsSource::get_(char* pvalS, int** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) get_(pArrDesc[p], &(*oparamVal)[p], isenum, oListLen);
}
void sParmsSource::get_(char* pvalS, bool** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) get_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParmsSource::get_(char* pvalS, numtype** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) get_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParmsSource::get_(char* pvalS, char** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) get_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
