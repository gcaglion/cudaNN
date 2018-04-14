#include "Debugger.h"

void sDebugger::sDebugger_common(int level_, int dest_, char* outFileName_, char* outFilePath_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_) {
	level=level_; dest=dest_; timing=timing_; PauseOnError=PauseOnError_; ThreadSafeLogging=ThreadSafeLogging_;
	try {
		outFile=new tFileInfo(outFileName_, outFilePath_, FILE_MODE_WRITE);
	}
	catch (std::exception e) {
		sprintf_s(errmsg, "Could not create Debugger outfile (%s/%s)", outFilePath_, outFileName_); throw std::runtime_error(errmsg);
	}
}
sDebugger::sDebugger(int level_, int dest_, char* outFileName_, char* outFilePath_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_) {
	sDebugger_common(level_, dest_, outFileName_, outFilePath_, timing_, PauseOnError_, ThreadSafeLogging_);
}
sDebugger::sDebugger(char* outFileName_) {
	sDebugger_common(DBG_LEVEL_DEFAULT, DBG_DEST_DEFAULT, outFileName_, DBG_DEFAULT_PATH, false, true, false);
}
sDebugger::sDebugger(int level_, int dest_, char* outFileName_) {
	sDebugger_common(level_, dest_, outFileName_, DBG_DEFAULT_PATH, false, false, false);
}
sDebugger::~sDebugger() {
	delete outFile;
}

//-- timing methods
void sDebugger::setStartTime() { startTime=timeGetTime(); }
void sDebugger::setElapsedTime() { elapsedTime=(DWORD)(timeGetTime()-startTime); }
//-- logging methods
void sDebugger::write(int LogType, const char* msg, int argcount, ...) {
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
void sDebugger::compose(char* msg_, int argcount, ...) {
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
	//errmsg[0]='\0';
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
template <typename T> void sDebugger::argOut(int msgType, char* submsg, T arg) {
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

