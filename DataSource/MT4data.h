#pragma once
#include "../CommonEnv.h"
#include "DataSource.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"

typedef struct sMT4Data : public sDataSource {
	int accountId;	//-- sarca cosa mi serve qui...

	EXPORT void sMT4Data_common(tDebugger* dbg_=nullptr);
	EXPORT sMT4Data(tParmsSource* parms, tDebugger* dbg_=nullptr);
	EXPORT sMT4Data(int accountId_=0, tDebugger* dbg_=nullptr);

} tMT4Data;
