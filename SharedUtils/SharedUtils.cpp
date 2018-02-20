#pragma once
#include "SharedUtils.h"

//-- generic (non-classed)
EXPORT void Trim(char* str) {
	int l = 0;
	int i;
	int r = (int)strlen(str);
	char ret[MAX_PATH];
	while (isspace(str[l])>0) l++;
	while (isspace(str[r-1])>0) r--;
	for (i = 0; i<(r-l); i++) ret[i] = str[l+i];
	ret[r-l] = '\0';
	//strcpy(str, ret);
	memcpy_s(str, strlen(str), ret, r-l-1);
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
int argcnt(const char* mask) {
	int cnt=0;
	for (int i=0; i<strlen(mask); i++) {
		if (mask[i]==37) cnt++;
	}
	return cnt;
}


sDbg::sDbg(int level_, int dest_, tFileInfo* outFile_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_) {
	level=level_; dest=dest_; timing=timing_; PauseOnError=PauseOnError_; ThreadSafeLogging=ThreadSafeLogging_;
	sprintf_s(stackmsg, 20, "\nError stack:");
	//-- outFile is created and opened by constructor (if not passed).
	if (outFile_==nullptr) {
		try {
			outFile=new tFileInfo("defaultDebug.log", DEBUG_DEFAULT_PATH);
		}
		catch (std::exception e) {
			sprintf_s(errmsg, 1024, "%s() error creating default debug file\nFrom: %s", __func__, e.what()); throw std::runtime_error(errmsg);
		}
	} else {
		outFile=outFile_;
	}
}

void sDbg::write(int LogType, const char* msg, int argcount, ...) {
	// pLogLevel=	0 (No screen, No file) | 1 (Screen, No file) | 2 (Screen AND File)
	int n;
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	va_list			arguments;
	char submsg[MAX_PATH];
	unsigned int	im = 0;
	int				prev_im = 0;

	if (LogType>level) return;

	if (ThreadSafeLogging) WaitForSingleObject(Mtx, INFINITE);

	va_start(arguments, argcount);
	n = 0;

	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			n++;
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				argOut(LogType, submsg, arg_s);
			} else if (submsg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				argOut(LogType, submsg, arg_d);
			} else if (submsg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				argOut(LogType, submsg, arg_d);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
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
void sDbg::compose(char* msg, int argcount, ...) {
	va_list			arguments;
	char submsg[MAX_PATH];
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	unsigned int	im = 0;
	int				prev_im = 0;

	va_start(arguments, argcount);
	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				sprintf_s(errmsg, submsg, arg_s); strcat_s(stackmsg, errmsg);
			} else if (submsg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				sprintf_s(errmsg, submsg, arg_d); strcat_s(stackmsg, errmsg);
			} else if (submsg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				sprintf_s(errmsg, submsg, arg_d); strcat_s(stackmsg, errmsg);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				sprintf_s(errmsg, submsg, arg_f); strcat_s(stackmsg, errmsg);
			}
		}
		im++;
	} while (im<strlen(msg));
	va_end(arguments);
}

template <typename T> void sDbg::argOut(int msgType, char* submsg, T arg) {
	if (msgType==DBG_LEVEL_ERR) {
		//-- file log is mandatory in case of error
		fprintf(outFile->handle, submsg, arg);
		//-- then, screen log only if defined by dest
		if (dest==DBG_DEST_SCREEN||dest==DBG_DEST_BOTH) printf(submsg, arg);
	} else {
		//-- check dest only
		if (dest==DBG_DEST_SCREEN||dest==DBG_DEST_BOTH) printf(submsg, arg);
		if (dest==DBG_DEST_FILE||dest==DBG_DEST_BOTH) fprintf(outFile->handle, submsg, arg);
	}
}

sFileInfo::sFileInfo(char* Name_, char* Path_, bool append_) {
	strcpy_s(Name, MAX_PATH, Name_);
	strcpy_s(Path, MAX_PATH, Path_);
	sprintf_s(creationTime, sizeof(creationTime), "%ld", timeGetTime());
	sprintf_s(FullName, MAX_PATH-1, "%s/%s_%s.log", Path, Name, creationTime);

	fopen_s(&handle, FullName, (append) ? "a" : "w");
	if (errno!=0) {
		sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d creating file %s", __func__, errno, FullName); throw std::runtime_error(errmsg);
	}
}
sFileInfo::~sFileInfo() {
	fclose(handle);
}

sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDebugInfo* DebugParms_) {
	if (DebugParms_==nullptr) {
		DebugParms=new tDebugInfo(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DBConnection.err"));
	} else {
		DebugParms=DebugParms_;
	}
	strcpy_s(DBUser, 30, username);
	strcpy_s(DBPassword, 30, password);
	strcpy_s(DBConnString, 30, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}

sFXData::sFXData(tDBConnection* db_, char* symbol_, char* tf_, int isFilled_) {
	db=db_;
	strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
	strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
	IsFilled=isFilled_;
}
