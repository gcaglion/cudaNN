#include "ParamMgr.h"

sParmsSource::sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("ParmsSource.err"))) : dbg_;
	CLoverridesCnt=CLoverridesCnt_; CLoverride=CLoverride_;
	safeCallEE(parmsFile=new tFileInfo(pFileFullName, FILE_MODE_READ));

	parmsCnt=0;
	currentKey[0]='\0';
	soughtKey[0]='\0';

	parmVal=(char***)malloc(XML_MAX_PARAMS_CNT*sizeof(char**)); 
	for (int p=0; p<XML_MAX_PARAMS_CNT; p++) {
		parmVal[p]=(char**)malloc(XML_MAX_ARRAY_PARAM_ELEM_CNT*sizeof(char*));
		for (int e=0; e<XML_MAX_ARRAY_PARAM_ELEM_CNT; e++) {
			parmVal[p][e]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
		}
	}
}
sParmsSource::~sParmsSource() {
	for (int p=0; p<XML_MAX_PARAMS_CNT; p++) {
		for (int e=0; e<XML_MAX_ARRAY_PARAM_ELEM_CNT; e++) {
			free(parmVal[p][e]);
		}
		free(parmVal[p]);
	}
	free(parmVal);
}
void sParmsSource::newDebugger(tDebugger* dbg_) {
	int dbg_level, dbg_dest; char dbg_fname[MAX_PATH];

	get(&dbg_level,"Level", false, enumlist);
	get(&dbg_dest, "Dest", false, enumlist);
	get(dbg_fname, "DestFileFullName");

	safeCallEE(dbg_=new tDebugger(dbg_level, dbg_dest, new tFileInfo(dbg_fname, FILE_MODE_WRITE)));

}

void stripLastStep(char* fullDesc, char* oStrippedDesc) {
	int lastDotPos=instr('.', fullDesc, true);
	//if (lastDotPos<0) return false;
	memcpy_s(oStrippedDesc, strlen(fullDesc), fullDesc, strlen(fullDesc)-lastDotPos-1);
	oStrippedDesc[lastDotPos]='\0';
}
bool sParmsSource::setKey(char* KeyDesc, bool fromRoot, bool ignoreError) {
	if (fromRoot) {
		currentKey[0]='\0';
	} else {
		strcat_s(currentKey, XML_MAX_PATH_LEN, ".");
	}
	strcat_s(currentKey, XML_MAX_PATH_LEN, KeyDesc);
	UpperCase(currentKey);

	bool found=findKey(currentKey);
	return(found);
}
bool sParmsSource::findKey(char* KeyFullDesc){
	char keyDescFromParmDesc[XML_MAX_PATH_LEN];
	for (int p=0; p<parmsCnt; p++) {
		stripLastStep(parmName[p], keyDescFromParmDesc);
		if (strcmp(keyDescFromParmDesc, KeyFullDesc)==0) return true;
	}
	return false;
}

//-- specific, single value: int(with or without enums), numtype, char*
/*void sParmsSource::getx(int* oVar, bool isenum, int* oArrLen) {
	if (isenum) {
		safeCallEB(decode(oVar));
	} else {
		(*oVar)=atoi(parmVal[foundParmId]);
	}
}
void sParmsSource::getx(bool* oVar, bool isenum, int* oArrLen) {
	(*oVar)=(strcmp(parmVal[foundParmId], "TRUE")==0);
}
void sParmsSource::getx(char* oVar, bool isenum, int* oArrLen) {
	strcpy_s(oVar, XML_MAX_PARAM_VAL_LEN, parmVal[foundParmId]);
}
void sParmsSource::getx(numtype* oVar, bool isenum, int* oArrLen) {
	(*oVar)=(numtype)atof(parmVal[foundParmId]);
}
*/
void sParmsSource::getx(int* oVar, bool isenum, int* oArrLen){
	getx(&oVar, isenum, oArrLen);
}
void sParmsSource::getx(bool* oVar, bool isenum, int* oArrLen){
	getx(&oVar, isenum, oArrLen);
}
void sParmsSource::getx(char* oVar, bool isenum, int* oArrLen){
	getx(&oVar, isenum, oArrLen);
}
void sParmsSource::getx(numtype* oVar, bool isenum, int* oArrLen){
	getx(&oVar, isenum, oArrLen);
}


//-- specific, arrays: int(with or without enums), numtype, char*
void sParmsSource::getx(int** oVar, bool isenum, int* oArrLen){
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		if (isenum) {
			safeCallEB(decode(e, &(*oVar[e])));
		} else {
			(*oVar[e])=atoi(parmVal[foundParmId][e]);
		}
	}
}
void sParmsSource::getx(bool** oVar, bool isenum, int* oArrLen) {
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		(*oVar[e])=(strcmp(parmVal[foundParmId][e], "TRUE")==0);
	}
}
void sParmsSource::getx(char** oVar, bool isenum, int* oArrLen){
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		strcpy_s(oVar[e], XML_MAX_PARAM_VAL_LEN, parmVal[foundParmId][e]);
	}
}
void sParmsSource::getx(numtype** oVar, bool isenum, int* oArrLen){
	for(int e=0; e<parmValsCnt[foundParmId]; e++){
		(*oVar[e])=(numtype)atof(parmVal[foundParmId][e]);
	}
}


bool sParmsSource::parse() {
	char vLine[1024];
	size_t llen;
	char readKeyDesc[XML_MAX_SECTION_DESC_LEN];
	char readParmDesc[XML_MAX_PARAM_NAME_LEN];
	char readParmVal[XML_MAX_PARAM_VAL_LEN];

	char fullKey[XML_MAX_PATH_LEN]="";

	rewind(parmsFile->handle);
	while (fgets(vLine, XML_MAX_LINE_SIZE, parmsFile->handle)!=NULL) {
		//-- strip spaces & tabs
		stripChar(vLine, ' ');
		stripChar(vLine, '\t');
		stripChar(vLine, '\n');
		llen=strlen(vLine);
		UpperCase(vLine);

		if (vLine[0]=='<' && vLine[1]!='/' && vLine[llen-1]=='>') {
			//-- key start
			memcpy_s(readKeyDesc, XML_MAX_SECTION_DESC_LEN, &vLine[1], llen-2); readKeyDesc[llen-2]='\0';
			if(strlen(fullKey)>0) strcat_s(fullKey, XML_MAX_PATH_LEN, ".");
			strcat_s(fullKey, XML_MAX_PATH_LEN, readKeyDesc);
		} else 	if (vLine[0]=='<' && vLine[1]=='/' && vLine[llen-1]=='>') {
			//-- key end
			memcpy_s(readKeyDesc, XML_MAX_SECTION_DESC_LEN, &vLine[2], llen-3); readKeyDesc[llen-3]='\0';
			//-- strip fullKey of the rightmost key
			stripLastStep(fullKey, fullKey);
		} else {
			//-- parameter
			if (!getValuePair(vLine, readParmDesc, readParmVal, '=')) return false;
			//-- add parameter full name to parmName[][]
			strcpy_s(parmName[parmsCnt], XML_MAX_PATH_LEN, fullKey);
			strcat_s(parmName[parmsCnt], XML_MAX_PATH_LEN, ".");
			strcat_s(parmName[parmsCnt], XML_MAX_PATH_LEN, readParmDesc);
			//-- also add parameter value[s] split array parameters
			parmValsCnt[parmsCnt]=cslToArray(readParmVal, ',', parmVal[parmsCnt]);
			parmsCnt++;
		}

	}


	return false;
}
