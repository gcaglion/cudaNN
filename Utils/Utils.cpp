#include "Utils.h"

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
EXPORT Bool getCurrentPath(char* oPath) {
	TCHAR Buffer[MAX_PATH];
	char  RetBuf[MAX_PATH];
	DWORD dwRet;
	size_t convcharsn;

	dwRet = GetCurrentDirectory(MAX_PATH, Buffer);
	if (dwRet==0) {
		printf("GetCurrentDirectory failed (%d)\n", GetLastError());
		return false;
	}
	wcstombs_s(&convcharsn, RetBuf, Buffer, MAX_PATH-1);
	strcpy_s(oPath, MAX_PATH, RetBuf);
	return true;
}
EXPORT void UpperCase(char* str) {
	int pos=0;
	while (str[pos]!='\0') {
		str[pos]=toupper(str[pos]);
		pos++;
	}
}
EXPORT int cslToArray(const char* csl, char Separator, char** StrList) {
	//-- 1. Put a <separator>-separated list of string values into an array of strings, and returns list length
	int i = 0;
	int prevSep = 0;
	int ListLen = 0;
	int kaz=0;

	while (csl[i]!='\0') {
		kaz = (prevSep==0) ? 0 : 1;
		if (csl[i]==Separator) {
			// separator
			memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
			StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
			stripChar(StrList[ListLen], ' '); stripChar(StrList[ListLen], '\t');
			if(strlen(StrList[ListLen])>0) ListLen++;
			prevSep = i;
		}
		i++;
	}
	//-- portion of pDesc after the last comma
	memcpy(StrList[ListLen], &csl[prevSep+kaz], i-prevSep-kaz);
	StrList[ListLen][i-prevSep-kaz] = '\0';	// add null terminator
	stripChar(StrList[ListLen], ' '); stripChar(StrList[ListLen], '\t');
	if (strlen(StrList[ListLen])==0) ListLen--;

	return (ListLen+1);
}
EXPORT void splitFullFileName(char* iFullName, char* oPath, char* oName) {
	int i;
	int ls=instr('\\', iFullName, true);
	if (ls<0) ls=instr('/', iFullName, true);

	oPath[0]='\0';
	for (i=0; i<ls; i++) oPath[i]=iFullName[i];
	oPath[i]='\0';

	oName[0]='\0';
	for (i=0; i<(strlen(iFullName)-ls); i++) oName[i]=iFullName[ls+i+1];
	oName[i]='\0';
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
EXPORT int  instr(char soughtChar, char* intoStr, Bool fromRight) {
	int i;
	if (fromRight) {
		for (i=(int)strlen(intoStr)-1; i>=0; i--) {
			if (intoStr[i]==soughtChar) return i;
		}
	} else {
		for (i=0; i<strlen(intoStr); i++) {
			if (intoStr[i]==soughtChar) return i;
		}
	}
	return -1;
}
int argcnt(const char* mask) {
	int cnt=0;
	for (int i=0; i<strlen(mask); i++) {
		if (mask[i]==37) cnt++;
	}
	return cnt;
}
EXPORT void removeQuotes(char* istr, char* ostr) {
	size_t slen=strlen(istr);
	size_t rlen=slen;
	int ri=0;
	for (int si=0; si<slen; si++) {
		if (istr[si]!=34) {
			ostr[ri]=istr[si];
			ri++;
		}
	}
	ostr[ri]='\0';
}
EXPORT void stripChar(char* istr, char c) {
	size_t ilen=strlen(istr);
	char* ostr=(char*)malloc(ilen+1);
	int ri=0;
	for (int si=0; si<ilen; si++) {
		if (istr[si]!=c) {
			ostr[ri]=istr[si];
			ri++;
		}
	}
	ostr[ri]='\0';
	memcpy_s(istr, ri, ostr, ri);
	istr[ri]='\0';
	free(ostr);
}
EXPORT Bool getValuePair(char* istr, char* oName, char* oVal, char eqSign) {
	int eqpos=instr(eqSign, istr, false); if (eqpos<0) return false;
	memcpy_s(oName, eqpos, istr, eqpos);
	oName[eqpos]='\0';
	memcpy_s(oVal, strlen(istr)-eqpos+1, &istr[eqpos+1], strlen(istr)-eqpos);
	oVal[strlen(istr)-eqpos]='\0';
	return true;
}
EXPORT Bool isnumber(char* str) {
	for (int i=0; i<strlen(str); i++) {
		if (str[i]>'9'||str[i]<'0') return false;
	}
	return true;
}
EXPORT Bool isInList(int soughtVal, int listLen, int* list) {
	for(int i=0; i<listLen; i++) {
		if (list[i]==soughtVal) return true;
	}
	return false;
}
