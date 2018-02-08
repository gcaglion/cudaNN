#pragma once

#include "../CommonEnv.h"
#include <vector>

#define LAYER_NN 0
#define LAYER_SVM 1
#define LAYER_GA 2
#define LAYER_SOM 3

class cEngine {
	int InputCount;
	int OutputCount;
	int layersCnt;
	cLayer** layer;

	cEngine(){
		layersCnt=0;
	}
	~cEngine(){
		for (int l=0; l<layersCnt; l++) delete layer[l];
	}

	int addLayer(int layerType, cLayer* layer_) {

	}

};

class cLayer {
public:
	int type;
	int level;
	int inputCnt;
	int outputCnt;

	std::vector<cLayer> parent;
	std::vector<cLayer> child;

	cLayer(cLayer* parent_, int inputCnt_, int outputCnt_) {
		if (parent_==NULL) {
			level=0;
		} else {
			level=parent_->level+1;
			parent.push_back(parent_);
			parent_->child.push_back(this);
		}
	}
	~cLayer() {}
private:
	cLayer(){}
};

class cNNlayer:cLayer {};
class cSVMlayer:cLayer {};