#include "MyDebug.h"

void getCurrentTime(char* ot) {
	time_t mytime = time(NULL);
	sprintf(ot, "%s", ctime(&mytime));
}

void sDebugInfo::write(int LogType, const char* msg, int argcount, ...) {
	// pLogLevel=	0 (No screen, No file) | 1 (Screen, No file) | 2 (Screen AND File)
	int n;
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	va_list			arguments;
	char submsg[MAX_PATH];
	unsigned int	im = 0;
	int				prev_im = 0;

	if (level==0&&LogType==DBG_INFO) return;

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
				if (level==1||level==3||LogType==DBG_ERROR) printf(submsg, arg_s);
				if (level==2||level==3||LogType==DBG_ERROR)	fprintf(outFile->handle, submsg, arg_s);
			} else if (msg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				if (level==1||level==3||LogType==DBG_ERROR) printf(submsg, arg_d);
				if (level==2||level==3||LogType==DBG_ERROR)	fprintf(outFile->handle, submsg, arg_d);
			} else if (msg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				if (level==1||level==3||LogType==DBG_ERROR) printf(submsg, arg_d);
				if (level==2||level==3||LogType==DBG_ERROR)	fprintf(outFile->handle, submsg, arg_d);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				if (level==1||level==3||LogType==DBG_ERROR) printf(submsg, arg_f);
				if (level==2||level==3||LogType==DBG_ERROR)	fprintf(outFile->handle, submsg, arg_f);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	if (level==1||level==3||LogType==DBG_ERROR) printf(submsg);
	if (level==2||level==3||LogType==DBG_ERROR) fprintf(outFile->handle, submsg);
	if (LogType==DBG_ERROR && PauseOnError) { printf("Press any key..."); getchar(); }

	va_end(arguments);

	if (ThreadSafeLogging) ReleaseMutex(Mtx);
}

sDebugInfo::sDebugInfo(int level_, char* fName_, char* fPath_, bool timing_, bool append_) {
	level=level_;  timing=timing_; 
	//-- setting defaults when. 
	ThreadSafeLogging=false;
	PauseOnError=true;

	if (level>0) {
		try {
			outFile=new tFileInfo(fPath_, fName_, append_);
		} catch (const char* e) { printf("Could not create debug file (%s). Exiting...\n", e); }
		char timestamp[60];	getCurrentTime(timestamp);
		fprintf(outFile->handle, "\n---------- Process %d Started New Log at %s ----------\n", GetCurrentProcessId(), timestamp);
	}

}
sDebugInfo::~sDebugInfo() {
	char timestamp[60];	getCurrentTime(timestamp);
	fprintf(outFile->handle, "\n---------- Process %d Closed current Log at %s ----------\n", GetCurrentProcessId(), timestamp);

	delete outFile;
}
