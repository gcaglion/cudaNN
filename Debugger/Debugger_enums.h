#pragma once

//-- DBG Level. Sets the level for the specific tDebugger* object
#define DBG_LEVEL_ERR 0	//-- errors
#define DBG_LEVEL_STD 1	//-- standard
#define DBG_LEVEL_DET 2	//-- detailed

//-- DBG Destination. Sets the destination for the specific tDebugger* object
#define DBG_DEST_SCREEN 0
#define DBG_DEST_FILE 1
#define DBG_DEST_BOTH 2

//-- default values for the two above
#define DBG_LEVEL_DEFAULT DBG_LEVEL_STD
#define DBG_DEST_DEFAULT DBG_DEST_BOTH
