#pragma once

#include "../CommonEnv.h"

#define CORE_MAX_DESC_LEN	128
#define CORE_MAX_PARENTS	32

#define coreCalled

typedef struct sCoreLayout : sBaseObj {
	int Id;
	char* desc;
	int layer;
	int type;
	int parentsCnt;
	int* parentId;
	int* parentConnType;

	int inputCnt;
	int outputCnt;

	EXPORT sCoreLayout();
	EXPORT ~sCoreLayout();

	EXPORT void addParent(int parentId_, int parentConnType_);

} tCoreLayout;

typedef struct sCore {

	tCoreLayout* layout;

	EXPORT sCore();
	EXPORT ~sCore();



} tCore;