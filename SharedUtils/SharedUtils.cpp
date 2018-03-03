#pragma once
#include "SharedUtils.h"

//=== generic (non-classed)
EXPORT char* MyGetCurrentDirectory() {
	TCHAR Buffer[MAX_PATH];
	char  RetBuf[MAX_PATH];
	DWORD dwRet;
	size_t convcharsn;

	dwRet = GetCurrentDirectory(MAX_PATH, Buffer);
	if (dwRet==0) {
		printf("GetCurrentDirectory failed (%d)\n", GetLastError());
	}
	wcstombs_s(&convcharsn, RetBuf, Buffer, MAX_PATH-1);
	return &RetBuf[0];
}
EXPORT void UpperCase(char* str) {
	int pos=0;
	while (str[pos]!='\0') {
		str[pos]=toupper(str[pos]);
		pos++;
	}
}
EXPORT void Trim(char* str) {
	int l = 0;
	int i;
	int r = (int)strlen(str);
	char ret[MAX_PATH];
	while (isspace(str[l])>0) l++;
	while (isspace(str[r-1])>0) r--;
	for (i = 0; i<(r-l); i++) ret[i] = str[l+i];
	ret[r-l] = '\0';
	strcpy(str, ret);
}
EXPORT int cslToArray(char* csl, char Separator, char** StrList) {
	//-- 1. Put a <separator>-separated list of string values into an array of strings, and returns list length
	int i = 0;
	int prevSep = 0;
	int ListLen = 0;
	int kaz;

	while (csl[i]!='\0') {
		kaz = (prevSep==0) ? 0 : 1;
		if (csl[i]==Separator) {
			// separator
			memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
			StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
			Trim(StrList[ListLen]);
			ListLen++;
			prevSep = i;
		}
		i++;
	}
	//-- portion of pDesc after the last comma
	memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
	StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
	Trim(StrList[ListLen]);

	return (ListLen+1);
}
EXPORT char* substr(char* str, int start, int len) {
	char ret[1000];
	memcpy(ret, &str[start], len);
	ret[len] = '\0';
	return &ret[0];
}
EXPORT char* right(char* str, int len) {
	return(substr(str, (int)strlen(str)-len, len));
}
EXPORT char* left(char* str, int len) {
	return(substr(str, 0, len));
}
int argcnt(const char* mask) {
	int cnt=0;
	for (int i=0; i<strlen(mask); i++) {
		if (mask[i]==37) cnt++;
	}
	return cnt;
}
void removeQuotes(char* istr, char* ostr) {
	size_t slen=strlen(istr);
	size_t rlen=slen;
	int ri=0;
	for (int si=0; si<slen; si++) {
		if(istr[si]!=34) {
			ostr[ri]=istr[si];
			ri++;
		}
	}
	ostr[ri]='\0';
}

//=== sDbg
sDbg::sDbg(int level_, int dest_, tFileInfo* outFile_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_) {
	level=level_; dest=dest_; timing=timing_; PauseOnError=PauseOnError_; ThreadSafeLogging=ThreadSafeLogging_;
	//-- outFile is created and opened by constructor (if not passed).
	if (outFile_==nullptr) {
		try {
			outFile=new tFileInfo("defaultDebug.log", DEBUG_DEFAULT_PATH);
		}
		catch (std::exception e) {
			sprintf_s(errmsg, sizeof(errmsg), "%s() error creating default debug file\nFrom: %s", __func__, e.what()); throw std::runtime_error(errmsg);
		}
	} else {
		outFile=outFile_;
	}
}
sDbg::~sDbg() {
	delete outFile;
}
//-- timing methods
void sDbg::setStartTime() { startTime=timeGetTime(); }
void sDbg::setElapsedTime() { elapsedTime=(DWORD)(timeGetTime()-startTime); }
//-- logging methods
void sDbg::write(int LogType, const char* msg, int argcount, ...) {
	if (LogType>level) return;

	char*			arg_s;
	int				arg_d;
	double			arg_f;
	va_list			arguments;
	char			submsg[1024];
	unsigned int	im=0, prev_im = 0;

	//--
	char fmask[16];
	int iim;
	//--

	if (ThreadSafeLogging) WaitForSingleObject(Mtx, INFINITE);

	va_start(arguments, argcount);
	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			if (msg[im+1]=='s') {   // "s"
				prev_im = im+2;
				arg_s = va_arg(arguments, char*);
				argOut(LogType, submsg, arg_s);
			} else if (msg[im+1]=='d') {   // "d"
				prev_im = im+2;
				arg_d = va_arg(arguments, int);
				argOut(LogType, submsg, arg_d);
			} else if (msg[im+1]=='p') {   // "p"
				prev_im = im+2;
				arg_d = va_arg(arguments, long);
				argOut(LogType, submsg, arg_d);
			} else {   // this could be 'f' or any mask before 'f' -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				//--
				iim=0;
				//-- if there's a mask before 'f', we need to re-define submsg
				while (msg[im+iim]!='f') {
					fmask[iim]=msg[im+iim];
					iim++;
				}
				fmask[iim]='f'; fmask[iim+1]='\0';
				memcpy(&submsg[strlen(submsg)-2], fmask, iim+2);
				im+=iim;
				prev_im=im+1;
				//--
				argOut(LogType, submsg, arg_f);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	argOut(LogType, submsg, arg_s);

	if (LogType==DBG_LEVEL_ERR && PauseOnError) { printf("Press any key..."); getchar(); }

	va_end(arguments);

	if (ThreadSafeLogging) ReleaseMutex(Mtx);
}
void sDbg::compose(char* msg_, int argcount, ...) {
	va_list			arguments;
	char submsg[MAX_PATH];
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	unsigned int	im = 0;
	int				prev_im = 0;
	char tmpmsg[1024];
	char msg[1024];

	va_start(arguments, argcount);
	removeQuotes(msg_, msg);
	errmsg[0]='\0';
	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				sprintf_s(tmpmsg, submsg, arg_s);
				strcat_s(errmsg, tmpmsg);
			} else if (submsg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				sprintf_s(tmpmsg, submsg, arg_d); strcat_s(errmsg, tmpmsg);
			} else if (submsg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				sprintf_s(tmpmsg, submsg, arg_d); strcat_s(errmsg, tmpmsg);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				sprintf_s(tmpmsg, submsg, arg_f); strcat_s(errmsg, tmpmsg);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	sprintf_s(tmpmsg, submsg, arg_s); strcat_s(errmsg, tmpmsg);

	va_end(arguments);
}
template <typename T> void sDbg::argOut(int msgType, char* submsg, T arg) {
	if (msgType==DBG_LEVEL_ERR) {
		//-- both file and screen log are mandatory in case of error
		fprintf(outFile->handle, submsg, arg);
		printf(submsg, arg);
	} else {
		//-- check dest only
		if (dest==DBG_DEST_SCREEN||dest==DBG_DEST_BOTH) printf(submsg, arg);
		if (dest==DBG_DEST_FILE||dest==DBG_DEST_BOTH) fprintf(outFile->handle, submsg, arg);
	}
}

//=== sFileInfo
sFileInfo::sFileInfo(char* Name_, char* Path_, int mode_) {
	strcpy_s(Name, MAX_PATH, Name_);
	strcpy_s(Path, MAX_PATH, Path_);
	sprintf_s(creationTime, sizeof(creationTime), "%ld", timeGetTime());
	sprintf_s(FullName, MAX_PATH-1, "%s/%s_%s.log", Path, Name, creationTime);

	setModeS(mode_); fopen_s(&handle, FullName, modeS);
	if (errno!=0) {
		sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d trying to %s file %s\n", __func__, errno, modeDesc, FullName); throw std::runtime_error(errmsg);
	}
}
sFileInfo::sFileInfo(char* FullName_, int mode_) {
	strcpy_s(FullName, MAX_PATH-1, FullName_);	//-- should also split Path/Name, and save them...

	setModeS(mode_); fopen_s(&handle, FullName, modeS);
	if (errno!=0) {
		sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d trying to %s file %s\n", __func__, errno, modeDesc, FullName); throw std::runtime_error(errmsg);
	}

}
sFileInfo::~sFileInfo() {
	fseek(handle, 0, SEEK_END); // seek to end of file
	size_t fsize = ftell(handle); // get current file pointer

	fclose(handle);

	if (fsize==0) remove(FullName);
}

void sFileInfo::setModeS(int mode_){
	switch (mode_) {
	case FILE_MODE_READ:
		strcpy_s(modeS, "r");
		strcpy_s(modeDesc, "Read"); break;
	case FILE_MODE_WRITE:
		strcpy_s(modeS, "w");
		strcpy_s(modeDesc, "Write"); break;
	case FILE_MODE_APPEND:
		strcpy_s(modeS, "a"); 
		strcpy_s(modeDesc, "Append"); break;
	default:
		sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d accessing file %s; invalid mode: (%d)\n", __func__, errno, FullName, mode_); throw std::runtime_error(errmsg);
		break;
	}
	mode=mode_;
}

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DBConnection.err"))) : dbg_;
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}
sDBConnection::~sDBConnection() { delete dbg; }

//=== sFXData
sFXData::sFXData(tDBConnection* db_, char* symbol_, char* tf_, int isFilled_) {
	db=db_;
	strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
	strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
	IsFilled=isFilled_;
}

//=== ParamMgr
sParamMgr::sParamMgr(tFileInfo* ParamFile_, int argc, char* argv[], tDbg* dbg_) {
	ParamFile=ParamFile_;
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("ParamMgr.err"))) : dbg_;

	//-- mallocs
	pArrDesc=(char**)malloc(ARRAY_PARAMETER_MAX_ELEMS*sizeof(char*)); for (int i=0; i<ARRAY_PARAMETER_MAX_ELEMS; i++) pArrDesc[i]=(char*)malloc(MAX_PARAMDESC_LEN);

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
	delete dbg;
}

//-- enums
void sParamMgr::getEnumVal(char* edesc, char* eVal, int* oVal) {
	if (strcmp(edesc, "FORECASTER.ACTION")==0) {
		//if (strcmp(eVal, "TRAIN_SAVE_RUN")==0) { (*oVal) = TRAIN_SAVE_RUN; return; }
		//if (strcmp(eVal, "ADD_SAMPLES")==0) { (*oVal) = ADD_SAMPLES; return; }
		//if (strcmp(eVal, "JUST_RUN")==0) { (*oVal) = JUST_RUN; return; }
/*	} else if (strcmp(edesc, "FORECASTER.ENGINE")==0) {
		if (strcmp(eVal, "ENGINE_NN")==0) { (*oVal) = ENGINE_NN; return; }
		if (strcmp(eVal, "ENGINE_GA")==0) { (*oVal) = ENGINE_GA; return; }
		if (strcmp(eVal, "ENGINE_SVM")==0) { (*oVal) = ENGINE_SVM; return; }
		if (strcmp(eVal, "ENGINE_SOM")==0) { (*oVal) = ENGINE_SOM; return; }
		if (strcmp(eVal, "ENGINE_WNN")==0) { (*oVal) = ENGINE_WNN; return; }
		if (strcmp(eVal, "ENGINE_XIE")==0) { (*oVal) = ENGINE_XIE; return; }
*/	} else if (strcmp(edesc, "RESULTS.DESTINATION")==0) {
		if (strcmp(eVal, "PERSIST_TO_TEXT")==0) { (*oVal) = PERSIST_TO_TEXT; return; }
		if (strcmp(eVal, "PERSIST_TO_ORCL")==0) { (*oVal) = PERSIST_TO_ORCL; return; }
/*	} else if (strcmp(edesc, "DATASOURCE.SOURCETYPE")==0) {
		if (strcmp(eVal, "SOURCE_DATA_FROM_FXDB")==0) { (*oVal) = SOURCE_DATA_FROM_FXDB; return; }
		if (strcmp(eVal, "SOURCE_DATA_FROM_FILE")==0) { (*oVal) = SOURCE_DATA_FROM_FILE; return; }
		if (strcmp(eVal, "SOURCE_DATA_FROM_MT")==0) { (*oVal) = SOURCE_DATA_FROM_MT; return; }
	} else if (strcmp(edesc, "DATASOURCE.TEXTFIELDSEPARATOR")==0) {
		if (strcmp(eVal, "COMMA")==0) { (*oVal) = (int)COMMA; return; }
		if (strcmp(eVal, "TAB")==0) { (*oVal) = (int)TAB; return; }
		if (strcmp(eVal, "SPACE")==0) { (*oVal) = (int)SPACE; return; }
*/	} else if (strcmp(edesc, "DATAPARMS.DATATRANSFORMATION")==0) {
		if (strcmp(eVal, "DT_NONE")==0) { (*oVal) = DT_NONE; return; }
		if (strcmp(eVal, "DT_DELTA")==0) { (*oVal) = DT_DELTA; return; }
		if (strcmp(eVal, "DT_LOG")==0) { (*oVal) = DT_LOG; return; }
		if (strcmp(eVal, "DT_DELTALOG")==0) { (*oVal) = DT_DELTALOG; return; }
/*	} else if (strcmp(edesc, "DATASOURCE.BARDATATYPES")==0) {
		if (strcmp(eVal, "OPEN")==0) { (*oVal) = OPEN; return; }
		if (strcmp(eVal, "HIGH")==0) { (*oVal) = HIGH; return; }
		if (strcmp(eVal, "LOW")==0) { (*oVal) = LOW; return; }
		if (strcmp(eVal, "CLOSE")==0) { (*oVal) = CLOSE; return; }
		if (strcmp(eVal, "VOLUME")==0) { (*oVal) = VOLUME; return; }
		if (strcmp(eVal, "OTHER")==0) { (*oVal) = OTHER; return; }
*/	} else if (strcmp(edesc, "DATAPARMS.TSFEATURES")==0) {
		if (strcmp(eVal, "TSF_MEAN")==0) { (*oVal) = TSF_MEAN; return; }
		if (strcmp(eVal, "TSF_MAD")==0) { (*oVal) = TSF_MAD; return; }
		if (strcmp(eVal, "TSF_VARIANCE")==0) { (*oVal) = TSF_VARIANCE; return; }
		if (strcmp(eVal, "TSF_SKEWNESS")==0) { (*oVal) = TSF_SKEWNESS; return; }
		if (strcmp(eVal, "TSF_KURTOSIS")==0) { (*oVal) = TSF_KURTOSIS; return; }
		if (strcmp(eVal, "TSF_TURNINGPOINTS")==0) { (*oVal) = TSF_TURNINGPOINTS; return; }
		if (strcmp(eVal, "TSF_SHE")==0) { (*oVal) = TSF_SHE; return; }
		if (strcmp(eVal, "TSF_HISTVOL")==0) { (*oVal) = TSF_HISTVOL; return; }
/*	} else if (strcmp(right(edesc, 7), "BP_ALGO")==0) {
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
*/	}

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
	throwE("ReadParamFromFile() could not find Parameter: %s . Exiting.\n", 1, pDesc);
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
	throwE("ReadParamFromFile() could not find Parameter: %s . Exiting.\n", 1, pDesc);
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
	throwE("ReadParamFromFile() could not find Parameter: %s . Exiting.\n", 1, pDesc);
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
	throwE("ReadParamFromFile() could not find Parameter: %s . Exiting.\n", 1, pDesc);
}
