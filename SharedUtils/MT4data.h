// File Retrieval Properties for File-based DataSources
#pragma once
#include "../CommonEnv.h"

typedef struct sMT4Data : public sDataSource {
	int accountId;	//-- sarca cosa mi serve qui...

	sMT4Data(int accountId_=0);

} tMT4Data;
