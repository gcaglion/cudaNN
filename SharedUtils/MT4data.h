#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "DataSource.h"
#include "DBConnection.h"

typedef struct sMT4Data : public sDataSource {
	int accountId;	//-- sarca cosa mi serve qui...

	void sMT4Data_common(tDebugger* dbg_=nullptr);
	EXPORT sMT4Data::sMT4Data(tParmsSource* parms, tDebugger* dbg_=nullptr);
	EXPORT sMT4Data(int accountId_=0, tDebugger* dbg_=nullptr);
	~sMT4Data() {}

} tMT4Data;
