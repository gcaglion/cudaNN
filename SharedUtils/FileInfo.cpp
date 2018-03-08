#include "FileInfo.h"

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
		sprintf_s(errmsg, sizeof(errmsg), "%s(): Error %d trying to %s file %s\n", __func__, errno, modeDesc, FullName);
		throw std::runtime_error(errmsg);
	}

}
sFileInfo::~sFileInfo() {
	fseek(handle, 0, SEEK_END); // seek to end of file
	size_t fsize = ftell(handle); // get current file pointer

	fclose(handle);

	if (fsize==0) remove(FullName);
}
void sFileInfo::setModeS(int mode_) {
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

