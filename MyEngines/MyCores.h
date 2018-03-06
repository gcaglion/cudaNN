#pragma once
#include "../CommonEnv.h"

//-- Limits
#define ENGINE_MAX_CORES 16

//-- Core types
#define CORE_NN 0
#define CORE_GA 1
#define CORE_SVM 2
#define CORE_SOM 3

//-- Core Connector types
#define CORE_CONN_DENSE  0
#define CORE_CONN_LINEAR 1

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