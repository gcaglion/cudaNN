#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"
#include <memory>

void Cleanup(int objCnt, ...) {
	va_list args;
	sBaseObj* obj;

	va_start(args, objCnt);
	for (int o=0; o<objCnt; o++) {
		obj=va_arg(args, sBaseObj*);
		delete (obj);
	}
	va_end(args);
}


int main() {
	
	tFileInfo* f0=nullptr;
	tFileInfo* f1=nullptr;
	tFileInfo* f2=nullptr;
	tFileInfo* f3=nullptr;
	tFileInfo* f4=nullptr;
	try {
		f0=new tFileInfo("C:/temp/logs/f0.log", FILE_MODE_WRITE);
		f1=new tFileInfo("f1.log", FILE_MODE_WRITE);
		f2=new tFileInfo("f2log", FILE_MODE_WRITE);
		f3=new tFileInfo("C:/temp/logs/f3log", FILE_MODE_WRITE);
		f4=new tFileInfo("C:\\temp\\logs\\f3log", FILE_MODE_WRITE);
	}
	catch (std::exception e) {
		printf("\nClient failed with exception: %s\n", e.what());
		Cleanup( 5, f0, f1, f2, f3, f4);
		system("pause");
		return -1;
	}

	Cleanup( 5, f0, f1, f2, f3, f4);
	system("pause");
	return 0;
}