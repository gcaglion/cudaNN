#include "ParamMgr.h"

sParmsSource::sParmsSource(char* pFileFullName, int CLoverridesCnt_, char* CLoverride_[], tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("ParmsSource.err"))) : dbg_;
	CLoverridesCnt=CLoverridesCnt_; CLoverride=CLoverride_;
	safeCallEE(parmsFile=new tFileInfo(pFileFullName, FILE_MODE_READ));

	currentKey=new tKey();
	soughtKey=new tKey();
	tmpKey=new tKey();
	currentParm=new tParm();
	soughtParm=new tParm();
	tmpParm=new tParm();

}

void sParmsSource::newDebugger(tDebugger* dbg_) {
	int dbg_level, dbg_dest; char dbg_fname[MAX_PATH];

	get(&dbg_level,"Level", false, enumlist);
	get(&dbg_dest, "Dest", false, enumlist);
	get(dbg_fname, "DestFileFullName");

	safeCallEE(dbg_=new tDebugger(dbg_level, dbg_dest, new tFileInfo(dbg_fname, FILE_MODE_WRITE)));

}

/*
//=== ParamMgr
sParamMgr::sParamMgr(tFileInfo* ParamFile_, int argc, char* argv[], tDebugger* dbg_) {
	ParamFile=ParamFile_;
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("ParamMgr.err"))) : dbg_;

	bool altIniFile = false;
	CLparamCount = argc;
	for (int p = 1; p < CLparamCount; p++) {
		char* pch = strchr(argv[p], '=');
		if (pch==NULL||argv[p][0]!='-'||argv[p][1]!='-') throwE("Command Line Fail on parameter %d !\nExiting...\n", 1, p);
		memcpy(&CLparamName[p][0], &argv[p][2], (pch-argv[p]-2)); CLparamName[p][pch-argv[p]-2] = '\0';
		memcpy(&CLparamVal[p][0], &argv[p][pch-argv[p]+1], strlen(pch));

		//-- if --IniFileName is specified, then set IniFileName global variable
		UpperCase(CLparamName[p]);
		if (strcmp(CLparamName[p], "INIFILE")==0) {
			altIniFile = true;
			safeCallEE(ParamFile=new tFileInfo(CLparamVal[p], FILE_MODE_READ));
		}
	}
	//-- if -- ParamFile is not passed, and IniFileName is not specified, then set look for default ini file in current directory
	if (!altIniFile && ParamFile==nullptr) {
		safeCallEE(ParamFile=new tFileInfo("Tester.ini", MyGetCurrentDirectory(), FILE_MODE_READ));
	}
}
sParamMgr::~sParamMgr() {
	delete dbg;
}
bool sParamMgr::sectionSet(const char* keyDesc, bool fromRoot, bool ignoreError) {
	soughtKey= new tXmlKey(keyDesc, fromRoot);
	bool found=soughtKey->find();
	if (!found&&!ignoreError) throwE("keyDesc=%s ; fromRoot=%s", 2, keyDesc, (fromRoot) ? "true" : "false");
	return found;
}

//--- create a new Debugger from ParmFile
void sParamMgr::newDebugger(tDebugger* dbg_) {
	int dbg_level, dbg_dest; char dbg_fname[MAX_PATH];

	getx(&dbg_level, "Level", enumlist);
	getx(&dbg_dest, "Dest", enumlist);
	getx(dbg_fname, "DestFileFullName");

	safeCallEE(dbg_=new tDebugger(dbg_level, dbg_dest, new tFileInfo(dbg_fname, FILE_MODE_WRITE)));

}

//=== XML stuff
void sXmlKey::sXmlKey_common() {
	step=(char**)malloc(XML_MAX_PATH_DEPTH*sizeof(char*)); for (int i=0; i<XML_MAX_PATH_DEPTH; i++) step[i]=(char*)malloc(XML_MAX_PARAM_NAME_LEN);
	full=(char*)malloc(XML_MAX_PATH_LEN);
	//-- init
	full[0]='\0';
	depth=0;
}
sXmlKey::sXmlKey() {
	sXmlKey_common();
}

sXmlKey::sXmlKey(const char* Key_, bool rootKey) {
	sXmlKey_common();
	if (rootKey) full[0]='\0';
	if (rootKey) {
		strcpy_s(full, XML_MAX_PATH_LEN, Key_);
	} else {
		if(strlen(full)>0) strcat_s(full, XML_MAX_PATH_LEN, ".");
		strcat_s(full, XML_MAX_PATH_LEN, Key_);
	}
	depth=cslToArray(full, '.', step);

}
sXmlKey::~sXmlKey() {
	for (int i=0; i<XML_MAX_PATH_DEPTH; i++) free(step[i]);
	free(step);
	free(full);
}
bool sXmlKey::find() {
	char keyStepTagStart[XML_MAX_SECTION_DESC_LEN+2];
	char keyStepTagEnd[XML_MAX_SECTION_DESC_LEN+3];
	bool startTagFound;

	//-- save current file pos
	fgetpos(parmFile, &parmFileKeyLocation);

	for (int d=0; d<depth; d++) {
		sprintf_s(keyStepTagStart, XML_MAX_SECTION_DESC_LEN+2, "<%s>", step[d]);
		sprintf_s(keyStepTagEnd, XML_MAX_SECTION_DESC_LEN+3, "</%s>", step[d]);

		//-- locate tag start
		startTagFound=false;
		while (fscanf(parmFile, "%s", keyStepTagStart)!=EOF) {
			if (strcmp(step[d], keyStepTagStart)==0) {
				startTagFound=true;
				break;
			}
		}
		//-- on failure, restore file pos
		if (!startTagFound) fsetpos(parmFile, &parmFileKeyLocation);
	}
	return (startTagFound);
}

//-- specific, single value: int(with or without enums), numtype, char*
void sParamMgr::getxx_(char* pvalS, int* oparamVal, bool isenum, int* oListLen) {
	if (isenum) {
		decode(pDesc, pvalS, oparamVal);
	} else {
		(*oparamVal)=atoi(pvalS);
	}
}
void sParamMgr::getxx_(char* pvalS, bool* oparamVal, bool isenum, int* oListLen) {
	Trim(pvalS); UpperCase(pvalS);
	(*oparamVal)=(strcmp(pvalS, "TRUE")==0);
}
void sParamMgr::getxx_(char* pvalS, numtype* oparamVal, bool isenum, int* oListLen) {
	(*oparamVal)=(numtype)atof(pvalS);
}
void sParamMgr::getxx_(char* pvalS, char* oparamVal, bool isenum, int* oListLen) {
	strcpy_s(oparamVal, XML_MAX_PARAM_VAL_LEN, pvalS);
}
//-- specific, arrays: int(with or without enums), numtype, char*
void sParamMgr::getxx_(char* pvalS, int** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], &(*oparamVal)[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, bool** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, numtype** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
void sParamMgr::getxx_(char* pvalS, char** oparamVal, bool isenum, int* oListLen) {
	(*oListLen)=cslToArray(pvalS, ',', pArrDesc);
	for (int p=0; p<(*oListLen); p++) getxx_(pArrDesc[p], oparamVal[p], isenum, oListLen);
}
*/