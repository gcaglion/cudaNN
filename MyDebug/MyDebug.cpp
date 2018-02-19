#include "MyDebug.h"

/*char* sDbg::getCurrTimeS() {
	sprintf_s(currTimeS, sizeof(currTimeS), "%ld", timeGetTime());
	return currTimeS;
}
*/

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

sDbg::sDbg(int level_, int dest_, tFileInfo* outFile_, bool timing_, bool PauseOnError_, bool ThreadSafeLogging_) {
	level=level_; dest=dest_; timing=timing_; PauseOnError=PauseOnError_; ThreadSafeLogging=ThreadSafeLogging_;
	//-- outFile is created and opened by constructor (if not passed).
	if (outFile_==nullptr) {
		try {
			outFile=new tFileInfo(DEBUG_DEFAULT_PATH, "defaultDebug.log");
		} catch (std::exception e) {
			sprintf_s(errmsg, 1024, "%s() error creating default debug file\nFrom: %s", __func__, e.what()); throw std::runtime_error(errmsg);
		}
	} else {
		outFile=outFile_;
	}
}