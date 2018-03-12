#pragma once
#include "../CommonEnv.h"

//-- Limits
#define ENGINE_MAX_CORES 16

typedef struct sCore {

	int Id;
	int type;
	int parentsCnt;
	int parentId[ENGINE_MAX_CORES];
	int parentConnType[ENGINE_MAX_CORES];

	//-- topology
	int InputCount;
	int OutputCount;

	sCore(){}

	sCore(int type_, int parentsCnt_, int* parentId_=nullptr, int* parentConnType_=nullptr){
		type=type_; parentsCnt=parentsCnt_;

		//-- define input & output dataSets. if parent is null, this core is located at level 0, else it's at parent level +1
		for (int p=0; p<parentsCnt; p++) {
			parentId[p]=parentId_[p];
			parentConnType[p]=parentConnType_[p];
		}

	}
} tCore;