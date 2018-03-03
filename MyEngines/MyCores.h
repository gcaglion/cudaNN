#pragma once
#include "../CommonEnv.h"

//-- Core types
#define CORE_NN 0
#define CORE_GA 1
#define CORE_SVM 2
#define CORE_SOM 3

typedef struct sCore {

	int type;

	//-- topology
	int InputCount;
	int OutputCount;
	sCore* parent;

	sCore(){}

	sCore(int type_, sCore* parent_=nullptr){
		type=type_; parent=parent_;
		//-- define input & output dataSets. if parent is null, this core is located at level 0, else it's at parent level +1
		if (parent==nullptr) {

		} else {

		}

	}
} tCore;