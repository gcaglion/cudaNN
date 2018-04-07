#include "Core.h"

sCoreLayout::sCoreLayout() {
	desc=(char*)malloc(CORE_MAX_DESC_LEN);
	parentId=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));
	parentConnType=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));
}
sCoreLayout::~sCoreLayout() {
	free(desc);
	free(parentId);
	free(parentConnType);
}

void sCoreLayout::addParent(int parentId_, int parentConnType_) {

}

sCore::sCore() {
	layout= new tCoreLayout();
}
sCore::~sCore(){
	delete layout;
}