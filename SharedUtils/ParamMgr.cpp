#include "ParamMgr.h"

//=== ParamMgr
sParamMgr::sParamMgr(tFileInfo* ParamFile_, int argc, char* argv[], tDbg* dbg_) {
	ParamFile=ParamFile_;
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("ParamMgr.err"))) : dbg_;

	//-- mallocs
	pArrDesc=(char**)malloc(ARRAY_PARAMETER_MAX_ELEMS*sizeof(char*)); for (int i=0; i<ARRAY_PARAMETER_MAX_ELEMS; i++) pArrDesc[i]=(char*)malloc(MAX_PARAMDESC_LEN);
	parmPath_Full=(char*)malloc(XML_MAX_PATH_LEN);
	parmPath_Step=(char**)malloc(XML_MAX_PATH_DEPTH*sizeof(char*)); for (int i=0; i<XML_MAX_PATH_DEPTH; i++) parmPath_Step[i]=(char*)malloc(XML_MAX_SECTION_DESC_LEN);
	parmDesc_Full=(char*)malloc(XML_MAX_PATH_LEN);
	parmDesc_Step=(char**)malloc(XML_MAX_PATH_DEPTH*sizeof(char*)); for (int i=0; i<XML_MAX_PATH_DEPTH; i++) parmDesc_Step[i]=(char*)malloc(XML_MAX_SECTION_DESC_LEN);

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
	for (int i=0; i<ARRAY_PARAMETER_MAX_ELEMS; i++) free(pArrDesc[i]);
	free(pArrDesc);
	for (int i=0; i<XML_MAX_PATH_DEPTH; i++) {
		free(parmPath_Step[i]);
		free(parmDesc_Step[i]);
	}
	free(parmPath_Step);
	free(parmPath_Full);
	free(parmDesc_Step);
	free(parmDesc_Full);
	delete dbg;
}

//=== XML stuff
void sParamMgr::sectionSet(const char* sectionLabel) {
	strcpy_s(parmPath_Full, XML_MAX_PATH_LEN, sectionLabel);
}
void sParamMgr::sectionSetChild(const char* child) {
	strcat_s(parmPath_Full, XML_MAX_PATH_LEN, "."); strcat_s(parmPath_Full, XML_MAX_PATH_LEN, child);
}
void sParamMgr::sectionSetParent() {
	//-- 1. delete current content
	strcpy_s(parmPath_Full, XML_MAX_PATH_LEN, "");
	//-- 2. add all but last path step
	for (int d=0; d<(parmDesc_depth-1); d++) {
		strcat(parmPath_Full, parmDesc_Step[d]); strcat(parmPath_Full, ".");
	}
	//-- 3. remove last "."
	parmPath_Full[strlen(parmPath_Full)-1]='\0';
}
void sParamMgr::enumDecode(char* pName, char* pVal, int* opvalIdx) {
	int decodedVal=-1;
	/*	if (strcmp(pName, "DATATRANSFORMATION")==0) {
	if (strcmp(pVal, "DT_NONE")==0) decodedVal=DT_NONE;
	if (strcmp(pVal, "DT_DELTA")==0) decodedVal=DT_DELTA;
	if (strcmp(pVal, "DT_LOG")==0) decodedVal=DT_LOG;
	if (strcmp(pVal, "DT_DELTALOG")==0) decodedVal=DT_DELTALOG;
	} else if (strcmp(pName, "STATISTICALFEATURES")==0) {
	if (strcmp(pVal, "TSF_MEAN")==0) decodedVal=TSF_MEAN;
	if (strcmp(pVal, "TSF_MAD")==0) decodedVal=TSF_MAD;
	if (strcmp(pVal, "TSF_VARIANCE")==0) decodedVal=TSF_VARIANCE;
	if (strcmp(pVal, "TSF_SKEWNESS")==0) decodedVal=TSF_SKEWNESS;
	if (strcmp(pVal, "TSF_KURTOSIS")==0) decodedVal=TSF_KURTOSIS;
	if (strcmp(pVal, "TSF_TURNINGPOINTS")==0) decodedVal=TSF_TURNINGPOINTS;
	if (strcmp(pVal, "TSF_SHE")==0) decodedVal=TSF_SHE;
	if (strcmp(pVal, "TSF_HISTVOL")==0) decodedVal=TSF_HISTVOL;
	} else if (strcmp(pName, "DATASOURCETYPE")==0) {
	if (strcmp(pVal, "SOURCE_DATA_FROM_FXDB")==0) decodedVal=SOURCE_DATA_FROM_FXDB;
	if (strcmp(pVal, "SOURCE_DATA_FROM_FILE")==0) decodedVal=SOURCE_DATA_FROM_FILE;
	if (strcmp(pVal, "SOURCE_DATA_FROM_MT4")==0) decodedVal=SOURCE_DATA_FROM_MT4;
	} else if (strcmp(pName, "SELECTEDFEATURES")==0) {
	if (strcmp(pVal, "OPEN")==0) decodedVal=OPEN;
	if (strcmp(pVal, "HIGH")==0) decodedVal=HIGH;
	if (strcmp(pVal, "LOW")==0) decodedVal=LOW;
	if (strcmp(pVal, "CLOSE")==0) decodedVal=CLOSE;
	if (strcmp(pVal, "VOLUME")==0) decodedVal=VOLUME;
	} else {

	}
	*/
	if (decodedVal==-1) {
		throwE("enumDecode() could not decode %s = %s", 2, pName, pVal);
	} else {
		(*opvalIdx)=decodedVal;
	}
}
//-- specific, single value: int(with or without enums), numtype, char*
void sParamMgr::getxx_(char* pvalS, int* oparamVal, bool isenum, int* oListLen) {
	if (isenum) {
		enumDecode(pDesc, pvalS, oparamVal);
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

//-- enumsDecode, OLD (pre-XML) 
/*void sParamMgr::getEnumVal(char* edesc, char* eVal, int* oVal) {
if (strcmp(edesc, "FORECASTER.ACTION")==0) {
//if (strcmp(eVal, "TRAIN_SAVE_RUN")==0) { (*oVal) = TRAIN_SAVE_RUN; return; }
//if (strcmp(eVal, "ADD_SAMPLES")==0) { (*oVal) = ADD_SAMPLES; return; }
//if (strcmp(eVal, "JUST_RUN")==0) { (*oVal) = JUST_RUN; return; }
} else if (strcmp(edesc, "FORECASTER.ENGINE")==0) {
if (strcmp(eVal, "ENGINE_NN")==0) { (*oVal) = ENGINE_NN; return; }
if (strcmp(eVal, "ENGINE_GA")==0) { (*oVal) = ENGINE_GA; return; }
if (strcmp(eVal, "ENGINE_SVM")==0) { (*oVal) = ENGINE_SVM; return; }
if (strcmp(eVal, "ENGINE_SOM")==0) { (*oVal) = ENGINE_SOM; return; }
if (strcmp(eVal, "ENGINE_WNN")==0) { (*oVal) = ENGINE_WNN; return; }
if (strcmp(eVal, "ENGINE_XIE")==0) { (*oVal) = ENGINE_XIE; return; }
} else if (strcmp(edesc, "RESULTS.DESTINATION")==0) {
if (strcmp(eVal, "TXT")==0) { (*oVal) = TXT; return; }
if (strcmp(eVal, "ORCL")==0) { (*oVal) = ORCL; return; }
} else if (strcmp(edesc, "DATASOURCE.SOURCETYPE")==0) {
if (strcmp(eVal, "SOURCE_DATA_FROM_FXDB")==0) { (*oVal) = SOURCE_DATA_FROM_FXDB; return; }
if (strcmp(eVal, "SOURCE_DATA_FROM_FILE")==0) { (*oVal) = SOURCE_DATA_FROM_FILE; return; }
if (strcmp(eVal, "SOURCE_DATA_FROM_MT")==0) { (*oVal) = SOURCE_DATA_FROM_MT; return; }
} else if (strcmp(edesc, "DATASOURCE.TXTFIELDSEPARATOR")==0) {
if (strcmp(eVal, "COMMA")==0) { (*oVal) = (int)COMMA; return; }
if (strcmp(eVal, "TAB")==0) { (*oVal) = (int)TAB; return; }
if (strcmp(eVal, "SPACE")==0) { (*oVal) = (int)SPACE; return; }
} else if (strcmp(edesc, "DATAPARMS.DATATRANSFORMATION")==0) {
if (strcmp(eVal, "DT_NONE")==0) { (*oVal) = DT_NONE; return; }
if (strcmp(eVal, "DT_DELTA")==0) { (*oVal) = DT_DELTA; return; }
if (strcmp(eVal, "DT_LOG")==0) { (*oVal) = DT_LOG; return; }
if (strcmp(eVal, "DT_DELTALOG")==0) { (*oVal) = DT_DELTALOG; return; }
} else if (strcmp(edesc, "DATASOURCE.BARDATATYPES")==0) {
if (strcmp(eVal, "OPEN")==0) { (*oVal) = OPEN; return; }
if (strcmp(eVal, "HIGH")==0) { (*oVal) = HIGH; return; }
if (strcmp(eVal, "LOW")==0) { (*oVal) = LOW; return; }
if (strcmp(eVal, "CLOSE")==0) { (*oVal) = CLOSE; return; }
if (strcmp(eVal, "VOLUME")==0) { (*oVal) = VOLUME; return; }
if (strcmp(eVal, "OTHER")==0) { (*oVal) = OTHER; return; }
} else if (strcmp(edesc, "DATAPARMS.TSFEATURES")==0) {
if (strcmp(eVal, "TSF_MEAN")==0) { (*oVal) = TSF_MEAN; return; }
if (strcmp(eVal, "TSF_MAD")==0) { (*oVal) = TSF_MAD; return; }
if (strcmp(eVal, "TSF_VARIANCE")==0) { (*oVal) = TSF_VARIANCE; return; }
if (strcmp(eVal, "TSF_SKEWNESS")==0) { (*oVal) = TSF_SKEWNESS; return; }
if (strcmp(eVal, "TSF_KURTOSIS")==0) { (*oVal) = TSF_KURTOSIS; return; }
if (strcmp(eVal, "TSF_TURNINGPOINTS")==0) { (*oVal) = TSF_TURNINGPOINTS; return; }
if (strcmp(eVal, "TSF_SHE")==0) { (*oVal) = TSF_SHE; return; }
if (strcmp(eVal, "TSF_HISTVOL")==0) { (*oVal) = TSF_HISTVOL; return; }
} else if (strcmp(right(edesc, 7), "BP_ALGO")==0) {
if (strcmp(eVal, "BP_STD")==0) { (*oVal) = BP_STD; return; }
if (strcmp(eVal, "BP_QING")==0) { (*oVal) = BP_QING; return; }
if (strcmp(eVal, "BP_SCGD")==0) { (*oVal) = BP_SCGD; return; }
if (strcmp(eVal, "BP_LM")==0) { (*oVal) = BP_LM; return; }
if (strcmp(eVal, "BP_QUICKPROP")==0) { (*oVal) = BP_QUICKPROP; return; }
if (strcmp(eVal, "BP_RPROP")==0) { (*oVal) = BP_RPROP; return; }
} else if (strcmp(right(edesc, 18), "ACTIVATIONFUNCTION")==0) {
if (strcmp(eVal, "NN_ACTIVATION_TANH")==0) { (*oVal) = NN_ACTIVATION_TANH; return; }
if (strcmp(eVal, "NN_ACTIVATION_EXP4")==0) { (*oVal) = NN_ACTIVATION_EXP4; return; }
if (strcmp(eVal, "NN_ACTIVATION_RELU")==0) { (*oVal) = NN_ACTIVATION_RELU; return; }
if (strcmp(eVal, "NN_ACTIVATION_SOFTPLUS")==0) { (*oVal) = NN_ACTIVATION_SOFTPLUS; return; }
} else if (strcmp(edesc, "SOMINFO.TDFUNCTION")==0) {
if (strcmp(eVal, "TD_DECAY_CONSTANT")==0) { (*oVal) = TD_DECAY_CONSTANT; return; }
if (strcmp(eVal, "TD_DECAY_LINEAR")==0) { (*oVal) = TD_DECAY_LINEAR; return; }
if (strcmp(eVal, "TD_DECAY_STEPPED")==0) { (*oVal) = TD_DECAY_STEPPED; return; }
if (strcmp(eVal, "TD_DECAY_EXP")==0) { (*oVal) = TD_DECAY_EXP; return; }
} else if (strcmp(edesc, "SOMINFO.LRFUNCTION")==0) {
if (strcmp(eVal, "LR_DECAY_CONSTANT")==0) { (*oVal) = LR_DECAY_CONSTANT; return; }
if (strcmp(eVal, "LR_DECAY_LINEAR")==0) { (*oVal) = LR_DECAY_LINEAR; return; }
if (strcmp(eVal, "LR_DECAY_STEPPED")==0) { (*oVal) = LR_DECAY_STEPPED; return; }
if (strcmp(eVal, "LR_DECAY_EXP")==0) { (*oVal) = LR_DECAY_EXP; return; }
} else if (strcmp(right(edesc, 10), "KERNELTYPE")==0) {
if (strcmp(eVal, "KERNEL_TYPE_LINEAR")==0) { (*oVal) = KERNEL_TYPE_LINEAR; return; }
if (strcmp(eVal, "KERNEL_TYPE_POLY")==0) { (*oVal) = KERNEL_TYPE_POLY; return; }
if (strcmp(eVal, "KERNEL_TYPE_RBF")==0) { (*oVal) = KERNEL_TYPE_RBF; return; }
if (strcmp(eVal, "KERNEL_TYPE_TANH")==0) { (*oVal) = KERNEL_TYPE_TANH; return; }
if (strcmp(eVal, "KERNEL_TYPE_CUSTOM")==0) { (*oVal) = KERNEL_TYPE_CUSTOM; return; }
} else if (strcmp(right(edesc, 9), "SLACKNORM")==0) {
if (strcmp(eVal, "SLACK_NORM_L1")==0) { (*oVal) = SLACK_NORM_L1; return; }
if (strcmp(eVal, "SLACK_NORM_SQUARED")==0) { (*oVal) = SLACK_NORM_SQUARED; return; }
} else if (strcmp(right(edesc, 15), "RESCALINGMETHOD")==0) {
if (strcmp(eVal, "RESCALING_METHOD_SLACK")==0) { (*oVal) = RESCALING_METHOD_SLACK; return; }
if (strcmp(eVal, "RESCALING_METHOD_MARGIN")==0) { (*oVal) = RESCALING_METHOD_MARGIN; return; }
} else if (strcmp(right(edesc, 12), "LOSSFUNCTION")==0) {
if (strcmp(eVal, "LOSS_FUNCTION_ZEROONE")==0) { (*oVal) = LOSS_FUNCTION_ZEROONE; return; }
} else if (strcmp(right(edesc, 12), "LEARNINGALGO")==0) {
if (strcmp(eVal, "LEARNING_ALGO_NSLACK")==0) { (*oVal) = LEARNING_ALGO_NSLACK; return; }
if (strcmp(eVal, "LEARNING_ALGO_NSLACK_SHRINK")==0) { (*oVal) = LEARNING_ALGO_NSLACK_SHRINK; return; }
if (strcmp(eVal, "LEARNING_ALGO_1SLACK_PRIMAL")==0) { (*oVal) = LEARNING_ALGO_1SLACK_PRIMAL; return; }
if (strcmp(eVal, "LEARNING_ALGO_1SLACK_DUAL")==0) { (*oVal) = LEARNING_ALGO_1SLACK_DUAL; return; }
if (strcmp(eVal, "LEARNING_ALGO_1SLACK_DUAL_CONSTR")==0) { (*oVal) = LEARNING_ALGO_1SLACK_DUAL_CONSTR; return; }
if (strcmp(eVal, "LEARNING_ALGO_CUSTOM")==0) { (*oVal) = LEARNING_ALGO_CUSTOM; return; }
}
throwE("getEnumVal() could not resolve Parameter: %s = %s . Exiting.\n", 2, edesc, eVal);

}

//-- single value (int, double, char*, enum) paramName should already be UpperCased & Trimmed
void sParamMgr::get_(numtype* oparamVal, bool isenum, int* oListLen) {
for (int p = 1; p < CLparamCount; p++) {
if (strcmp(CLparamName[p], pDesc)==0) {
(*oparamVal) = (numtype)atof(CLparamVal[p]);
return;
}
}
safeCallEE(ReadParamFromFile(oparamVal));
}
void sParamMgr::get_(char* oparamVal, bool isenum, int* oListLen) {
for (int p = 1; p < CLparamCount; p++) {
if (strcmp(CLparamName[p], pDesc)==0) {
strcpy_s(oparamVal, MAX_PARAMDESC_LEN, CLparamVal[p]);
return;
}
}
safeCallEE(ReadParamFromFile(oparamVal));
}
void sParamMgr::get_(int* oparamVal, bool isenum, int* oListLen) {
char evals[100];
int ret = 0;
for (int p = 1; p < CLparamCount; p++) {
if (strcmp(CLparamName[p], pDesc)==0) {
strcpy_s(evals, MAX_PARAMDESC_LEN, CLparamVal[p]);
safeCallEE(getEnumVal(pDesc, evals, oparamVal));
return;
}
}
if (isenum) {
safeCallEE(ReadParamFromFile(evals));
safeCallEE(getEnumVal(pDesc, evals, oparamVal));
} else {
safeCallEE(ReadParamFromFile(oparamVal));
}
}
void sParamMgr::get_(bool* oparamVal, bool isenum, int* oListLen) {
for (int p = 1; p < CLparamCount; p++) {
if (strcmp(CLparamName[p], pDesc)==0) {
(*oparamVal) = (strcmp(CLparamVal[p],"TRUE")==0);
return;
}
}
safeCallEE(ReadParamFromFile(oparamVal));
}
//-- array values
void sParamMgr::get_(numtype** oparamVal, bool isenum, int* oListLen) {
//-- first, get the list as a regular char* parameter
get_(pListDesc);
//-- then, split
(*oListLen) = cslToArray(pListDesc, ',', pArrDesc);
//-- finally, convert
for (int i=0; i<(*oListLen); i++) (*oparamVal)[i] = (numtype)atof(pArrDesc[i]);
}
void sParamMgr::get_(char** oparamVal, bool isenum, int* oListLen) {
//-- first, get the list as a regular char* parameter
get_(pListDesc);
//-- then, split
(*oListLen) = cslToArray(pListDesc, ',', pArrDesc);
//-- finally, convert
for (int i=0; i<(*oListLen); i++) strcpy(oparamVal[i], pArrDesc[i]);
}
void sParamMgr::get_(int** oparamVal, bool isenum, int* oListLen) {
//-- first, get the list as a regular char* parameter
get_(pListDesc);
//-- then, split
(*oListLen) = cslToArray(pListDesc, ',', pArrDesc);
//-- for each element, check if we need enum
for (int i=0; i<(*oListLen); i++){
if (isenum) {
safeCallEE(getEnumVal(pDesc, pArrDesc[i], &(*oparamVal)[i]));
} else {
//-- convert
for (int i=0; i<(*oListLen); i++) (*oparamVal)[i] = atoi(pArrDesc[i]);
}
}
}

void sParamMgr::ReadParamFromFile(int* oParamValue) {
char vParamName[1000];
char vParamValue[1000];

rewind(ParamFile->handle);
while (fscanf(ParamFile->handle, "%s = %s ", &vParamName[0], &vParamValue[0])!=EOF) {
Trim(vParamName); UpperCase(vParamName);
if (strcmp(&vParamName[0], pDesc)==0) {
(*oParamValue) = atoi(vParamValue);
return;
}
}
throwE("Could not find Parameter: %s", 1, pDesc);
}
void sParamMgr::ReadParamFromFile(numtype* oParamValue) {
char vParamName[1000];
char vParamValue[1000];

rewind(ParamFile->handle);
while (fscanf(ParamFile->handle, "%s = %s ", &vParamName[0], &vParamValue[0])!=EOF) {
Trim(vParamName); UpperCase(vParamName);
if (strcmp(&vParamName[0], pDesc)==0) {
(*oParamValue) = (numtype)atof(vParamValue);
return;
}
}
throwE("Could not find Parameter: %s", 1, pDesc);
}
void sParamMgr::ReadParamFromFile(char* oParamValue) {
char vParamName[1000];
char vParamValue[1000];

rewind(ParamFile->handle);
while (fscanf(ParamFile->handle, "%s = %[^\n]", &vParamName[0], &vParamValue[0])!=EOF) {
Trim(vParamName); UpperCase(vParamName);
if (strcmp(&vParamName[0], pDesc)==0) {
strcpy_s(oParamValue, MAX_PARAMDESC_LEN, vParamValue);
return;
}
}
throwE("Could not find Parameter: %s", 1, pDesc);
}
void sParamMgr::ReadParamFromFile(bool* oParamValue) {
char vParamName[1000];
char vParamValue[1000];

rewind(ParamFile->handle);
while (fscanf(ParamFile->handle, "%s = %s ", &vParamName[0], &vParamValue[0])!=EOF) {
Trim(vParamName); UpperCase(vParamName);
if (strcmp(&vParamName[0], pDesc)==0) {
Trim(vParamValue); UpperCase(vParamValue);
(*oParamValue) = (strcmp(vParamValue, "TRUE")==0);
return;
}
}
throwE("Could not find Parameter: %s", 1, pDesc);
}
*/

