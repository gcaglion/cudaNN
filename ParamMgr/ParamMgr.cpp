#include "ParamMgr.h"

sParmsSource::sParmsSource(char* objName_, s0* objParent_, char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], Bool autoParse, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {

	CLoverridesCnt=CLoverridesCnt_; CLoverride=CLoverride_;

	spawnFile(parmsFile, pFileFullName, FILE_MODE_READ);

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

	if (autoParse) parse();
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
void sParmsSource::buildSoughtParmFull(const char* soughtParmDesc) {
	if (soughtParmDesc[0]=='.'||strlen(currentKey)==0) {
		soughtKey[0]='\0';
	} else {
		strcpy_s(soughtKey, XML_MAX_PATH_LEN, currentKey);
		strcat_s(soughtKey, ".");
	}
	strcpy_s(soughtParmFull, XML_MAX_PATH_LEN, soughtKey);
	strcat_s(soughtParmFull, soughtParmDesc);
	UpperCase(soughtParmFull);
}
int sParmsSource::findParmId(){
	int ret=-1;
	for (int p=0; p<parmsCnt; p++) {
		if (strcmp(soughtParmFull, parmName[p])==0) {
			ret=p;
			break;
		}
	}
	return ret;
}

Bool stripLastStep(char* fullDesc, char* oStrippedDesc) {
	int lastDotPos=instr('.', fullDesc, true);
	if (lastDotPos<0) return false;
	memcpy_s(oStrippedDesc, strlen(fullDesc), fullDesc, lastDotPos);
	oStrippedDesc[lastDotPos]='\0';
	return true;
}

void sParmsSource::setKey(char* KeyDesc_, Bool ignoreError) {
	
	//-- KeyDesc may be passed as literal, therefore we need a buffer to copy KeyDesc_ to, so we can overwrite it
	char KeyDesc[XML_MAX_PATH_LEN];	strcpy_s(KeyDesc, XML_MAX_PATH_LEN, KeyDesc_);
	//-- save current key, to be printed in case of failure
	char tmpCurrentKey[XML_MAX_PATH_LEN]; strcpy_s(tmpCurrentKey, XML_MAX_PATH_LEN, currentKey);

	//-- "." before KeyDesc makes search start from root;
	if (KeyDesc[0]=='.' && KeyDesc[1]!='.') {
		currentKey[0]='\0';
		memcpy_s(KeyDesc, XML_MAX_PATH_LEN, &KeyDesc[1], XML_MAX_PATH_LEN-1);
		setKey(KeyDesc, ignoreError);
		return;
	}

	//-- ".." before KeyDesc makes search start from one level up;
	if (KeyDesc[0]=='.' && KeyDesc[1]=='.') {
		stripLastStep(currentKey, currentKey);
		memcpy_s(KeyDesc, XML_MAX_PATH_LEN, &KeyDesc[2], XML_MAX_PATH_LEN-2);
		setKey(KeyDesc, ignoreError);
		return;
	}

	//--  otherwise search starts from current pos
	if(strlen(currentKey)>0) strcat_s(currentKey, XML_MAX_PATH_LEN, ".");
	strcat_s(currentKey, XML_MAX_PATH_LEN, KeyDesc);

	UpperCase(currentKey);

	Bool success=(findKey(currentKey) || ignoreError);
	if (!success) {
		fail("CurrentKey=%s", KeyDesc_);
	}
}
Bool sParmsSource::findKey(char* KeyFullDesc){
	char keyDescFromParmDesc[XML_MAX_PATH_LEN];
	for (int p=0; p<parmsCnt; p++) {
		strcpy_s(keyDescFromParmDesc, XML_MAX_PATH_LEN, parmName[p]);
		while (stripLastStep(keyDescFromParmDesc, keyDescFromParmDesc)) {
			if (strcmp(keyDescFromParmDesc, KeyFullDesc)==0) return true;
		}
	}
	return false;
}
void sParmsSource::backupKey() {
	strcpy_s(bkpKey, XML_MAX_PATH_LEN, currentKey);
}
void sParmsSource::restoreKey() {
	strcpy_s(currentKey, XML_MAX_PATH_LEN, bkpKey);
}
//-- specific, single value: int(with or without enums), numtype, char*
void sParmsSource::getx(int* oVar){
	getx(&oVar);
}
void sParmsSource::getx(Bool* oVar){
	getx(&oVar);
}
void sParmsSource::getx(char* oVar){
	getx(&oVar);
}
void sParmsSource::getx(numtype* oVar){
	getx(&oVar);
}

//-- specific, arrays: int(with or without enums), numtype, char*
void sParmsSource::getx(int** oVar){
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		if (isnumber(parmVal[foundParmId][e])) {
			(*oVar)[e]=atoi(parmVal[foundParmId][e]);
		} else {
			safecall(decode(e, &oVar[0][e]));
		}
	}
}
void sParmsSource::getx(Bool** oVar) {
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		UpperCase(parmVal[foundParmId][e]);
		(*oVar)[e]=(strcmp(parmVal[foundParmId][e], "TRUE")==0);
	}
}
void sParmsSource::getx(char** oVar){
	for (int e=0; e<parmValsCnt[foundParmId]; e++) {
		for (int i=0; i<strlen(oVar[e]); i++) oVar[e][i]=parmVal[foundParmId][e][i];
		//memcpy_s(oVar[e], strlen(oVar[e])+1, parmVal[foundParmId][e], XML_MAX_PARAM_VAL_LEN);
		//strcpy_s(oVar[e], strlen(oVar[e])+1, parmVal[foundParmId][e]);
		//strcpy(oVar[e], parmVal[foundParmId][e]);
		//strcpy_s(oVar[e], XML_MAX_PARAM_VAL_LEN, parmVal[foundParmId][e]);
	}
}
void sParmsSource::getx(numtype** oVar){
	for(int e=0; e<parmValsCnt[foundParmId]; e++){
		(*oVar)[e]=(numtype)atof(parmVal[foundParmId][e]);
	}
}


Bool sParmsSource::parse() {
	char vLine[XML_MAX_LINE_SIZE];
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
		if (llen==0) continue;
		//UpperCase(vLine);

		if (vLine[0]=='<' && vLine[1]!='/' && vLine[llen-1]=='>') {
			//-- key start
			memcpy_s(readKeyDesc, XML_MAX_SECTION_DESC_LEN, &vLine[1], llen-2); readKeyDesc[llen-2]='\0';
			UpperCase(readKeyDesc);
			if(strlen(fullKey)>0) strcat_s(fullKey, XML_MAX_PATH_LEN, ".");
			strcat_s(fullKey, XML_MAX_PATH_LEN, readKeyDesc);
		} else 	if (vLine[0]=='<' && vLine[1]=='/' && vLine[llen-1]=='>') {
			//-- key end
			memcpy_s(readKeyDesc, XML_MAX_SECTION_DESC_LEN, &vLine[2], llen-3); readKeyDesc[llen-3]='\0';
			UpperCase(readKeyDesc);
			//-- strip fullKey of the rightmost key
			stripLastStep(fullKey, fullKey);
		} else {
			//-- parameter
			if (!getValuePair(vLine, readParmDesc, readParmVal, '=')) return false;
			UpperCase(readParmDesc);
			//-- add parameter full name to parmName[][]
			strcpy_s(parmName[parmsCnt], XML_MAX_PATH_LEN, fullKey);
			strcat_s(parmName[parmsCnt], XML_MAX_PATH_LEN, ".");
			strcat_s(parmName[parmsCnt], XML_MAX_PATH_LEN, readParmDesc);
			//-- also add parameter value[s] split array parameters
			parmValsCnt[parmsCnt]=cslToArray(readParmVal, ',', parmVal[parmsCnt]);
			parmsCnt++;
		}

	}
	return true;
}

/*
void sParmsSource::ObjectDebugParmsOverride(char* KeyDesc_, sDebugger* objdbg_) {
	
	setKey(KeyDesc_);

	setKey("Debugger", true);

	sDebuggerParms* newparms= new sDebuggerParms(objdbg_->parms->dest, objdbg_->parms->verbose, objdbg_->parms->timing, objdbg_->parms->pauseOnError, objdbg_->parms->outFileFullName, objdbg_->parms->outFilePath, objdbg_->parms->outFileName);

	//-- we need to ignore exceptions, so include all calls in try/catch
	try {
		safecall(get(&newparms->verbose, "Verbose"));
		safecall(get(&newparms->timing, "Timing"));
		safecall(get(&newparms->pauseOnError, "PauseOnError"));
		safecall(get(&newparms->dest, "Dest"));
		safecall(get(newparms->outFileFullName, "OutFileFullName"));
	}
	catch (std::exception exc) {
		//.. do nothing...
	}
	
	//-- delete current dbg
	delete objdbg_;
	//-- create new dbg with newparms, and assign it to objdbg_
	try {
		objdbg_=new sDebugger(newparms);
	}
	catch (std::exception exc) {
		char msg[DBG_MSG_MAXLEN]="";
		err("%s(%p)->%s() failed. Exception: %s", objName, this, __func__, exc.what());
		throw(exc);
	}
}
*/