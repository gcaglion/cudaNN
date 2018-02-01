#pragma once

//-- TimeSeries Statistical Features
#define MAX_TSFCOUNT 12
#define TSF_MEAN 0
#define TSF_MAD 1
#define TSF_VARIANCE 2
#define TSF_SKEWNESS 3
#define TSF_KURTOSIS 4
#define TSF_TURNINGPOINTS 5
#define TSF_SHE 6
#define TSF_HISTVOL 7

typedef struct {
	double Data;
	double Data_S;
	double ScaleMin;
	double ScaleMax;
	double ScaleM;
	double ScaleP;
	double* kaz;
} tTSF;
