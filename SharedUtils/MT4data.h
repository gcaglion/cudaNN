#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "DataSource.h"
#include "DBConnection.h"

typedef struct sMT4Data : public sDataSource {
	int accountId;	//-- sarca cosa mi serve qui...

	void sMT4Data_common(tDbg* dbg_=nullptr);
	EXPORT sMT4Data::sMT4Data(tParamMgr* parms, tDbg* dbg_=nullptr);
	EXPORT sMT4Data(int accountId_=0, tDbg* dbg_=nullptr);
	~sMT4Data() {}

} tMT4Data;
