#include <math.h>

#include "../CommonEnv.h"
#include "../MyDebug/MyDebug.h"
#include "../MyUtils/MyUtils.h"
#include "../OraUtils/OraUtils.h"
#include "TSF.h"
#include "../DataShape.h"
#include "../FXData.h"
#include "../FileData.h"

#define SOURCE_DATA_FROM_FXDB 0
#define SOURCE_DATA_FROM_FILE 1
#define SOURCE_DATA_FROM_MT   2

//-- Data Tranformations
#define DT_NONE		 0
#define DT_DELTA	 1
#define DT_LOG		 2
#define DT_DELTALOG	 3

//-- Latest Stuff
EXPORT int LoadFXdata(tDebugInfo* DebugParms, char* symbol, char* tf, char* date0, int seqlen, numtype* seqdata, numtype* baseBar);
EXPORT void SlideArrayF(int iWholeSetLen, numtype* iWholeSet, int featuresCnt, int iSampleCount, int iSampleSize, numtype** oSample, int iTargetSize, numtype** oTarget, int pWriteLog);
EXPORT void fSlideArrayF(int iWholeSetLen, numtype* iWholeSet, int featuresCnt, int iSampleCount, int iSampleSize, numtype* ofSample, int iTargetSize, numtype* ofTarget, int pWriteLog);
EXPORT void dataTrS(int datalen, int featuresCnt, numtype* Idata, numtype* base, int txtype, numtype scaleMin, numtype scaleMax, numtype* Odata, numtype* OscaleM, numtype* OscaleP);
//--

EXPORT int LoadData_FXDB(tDebugInfo* DebugParms, tFXData* DBParms, int pHistoryLen, int pFutureLen, char* pDate0, int pValidationShift, int pDatasetCount, numtype** oHistoryData, numtype** oHistoryBarW, numtype** oValidationData, numtype** oFutureData, numtype** oFutureBarW, numtype** oWholeData, numtype** oWholeBarW, numtype* oPrevValH, numtype* oPrevValV, numtype* oPrevBarW);
EXPORT int LoadData_CSV(tDebugInfo* DebugParms, tFileData* pDataFile, int pHistoryLen, int pFutureLen, char* pDate0, int pValidationShift, int pDatasetCount, numtype** oHistoryData, numtype** oHistoryBarW, numtype** oValidationData, numtype** oFutureData, numtype** oFutureBarW, numtype** oWholeData, numtype** oWholeBarW, numtype* oPrevValH, numtype* oPrevValV, numtype* oPrevBarW);
EXPORT int GetDates_CSV(tDebugInfo* DebugParms, tFileData* pDataFile, char* StartDate, int DatesCount, char** oDate);
EXPORT int GetDates_FXDB(tDebugInfo* DebugParms, tFXData* SourceParms, char* StartDate, int DatesCount, char** oDate);
EXPORT int LoadHistoryData(int pHistoryLen, char* pDate0, int pBarDataType, numtype* oHistoryData, tFXData* DBParms, tDebugInfo* DebugParms);

EXPORT void dataTransform(int dt, int dlen, numtype* idata, numtype baseVal, numtype* odata);
EXPORT void dataUnTransform(int dt, int dlen, int from_i, int to_i, numtype* idata, numtype baseVal, numtype* iActual, numtype* odata);
//-----------------------------------------------------------------------------------------------------------------------------------
EXPORT void TSfromSamples(int sampleCnt, int sampleLen, numtype** iSample, numtype* oTS);
EXPORT int FXDataCompact(char* INfilename, int INtimeframe, char* OUTfilename, int OUTtimeframe);

//-- Timeseries Statistical Features
EXPORT numtype TSMean(int VLen, numtype* V);
EXPORT numtype TSMeanAbsoluteDeviation(int VLen, numtype* V);
EXPORT numtype TSVariance(int VLen, numtype* V);
EXPORT numtype TSSkewness(int VLen, numtype* V);
EXPORT numtype TSKurtosis(int VLen, numtype* V);
EXPORT numtype TSTurningPoints(int VLen, numtype* V);
EXPORT numtype TSShannonEntropy(int VLen, numtype* V);
EXPORT numtype TSHistoricalVolatility(int VLen, numtype* V);
EXPORT void   CalcTSF(int TSFcnt, int* TSFid, tDataShape* pDataParms, int pTSLen, numtype* pTS, numtype* pTSF);
