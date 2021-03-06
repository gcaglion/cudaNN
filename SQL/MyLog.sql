-- connect system/manager@Algo
--create user cuLogUser identified by LogPwd default tablespace LogData;
--grant dba to cuLogUser;

--connect LogUser/LogPwd@Algo

--select table_name,constraint_name from user_constraints where constraint_type='R' order by 1,2;
----------------------------------------------------------------------------
alter table TradeInfo drop constraint TradeInfo_FK_ClientInfo;
alter table Dataparms drop constraint DataParms_FK_ClientInfo;
alter table Engineparms drop constraint EngineParms_FK_ClientInfo;
alter table EngineThreads drop constraint EngineThreads_FK_EngineParms;
alter table EngineThreads drop constraint EngineThreads_FK_DataParms;
alter table CoreImage_NN drop constraint CoreImage_NN_FK_EngineThreads;
alter table CoreImage_SOM drop constraint CoreImage_SOM_FK_EngineThreads;
alter table CoreImage_SVM drop constraint CoreImage_SVM_FK_EngineThreads;
--alter table CoreParms_NN drop constraint CoreParms_NN_FK_EngineThreads;
--alter table CoreParms_GA drop constraint CoreParms_GA_FK_EngineThreads;
--alter table CoreParms_SOM drop constraint CoreParms_SOM_FK_EngineThreads;
--alter table CoreParms_SVM drop constraint CoreParms_SVM_FK_EngineThreads;
alter table MSELog drop constraint MSELog_FK_EngineThreads;
alter table RunLog drop constraint RunLog_FK_EngineThreads;
alter table CoreLogs_NN drop constraint CoreLogs_NN_FK_EngineThreads;
alter table CoreLogs_NN_SCGD drop constraint CoreLogs_NN_SCGD_FK_ET;
alter table CoreLogs_GA drop constraint CoreLogs_GA_FK_EngineThreads;
alter table CoreLogs_SVM drop constraint CoreLogs_SVM_FK_EngineThreads;
alter table CoreLogs_SOM drop constraint CoreLogs_SOM_FK_EngineThreads;
----------------------------------------------------------------------------

drop table TradeInfo purge;
create table TradeInfo(
	ProcessId number,
	BarId number,
	LastBarT date,
	LastBarO number,
	LastBarH number,
	LastBarL number,
	LastBarC number,
	FirstBarT date,
	FirstBarO number,
	FirstBarH number,
	FirstBarL number,
	FirstBarC number,
	PrevFH number,
	PrevFL number,
	CurrBid number,
	CurrAsk number,
	CurrFH number,
	CurrFL number,
	TradeType number,
	TradeSize number,
	TradeTP number,
	TradeSL number
);
alter table TradeInfo add constraint TradeInfo_PK primary key(ProcessId, BarId) using index tablespace LogIdx;

drop table DataParms purge;
create table DataParms(
	ProcessId number,
	DatasetId number,
	DataSourceType number,
	DataSourceFileName varchar2(100),
	Symbol varchar2(8),
	TimeFrame varchar2(5),
	IsFilled number,
	BarData number,
	DataTransformation number,
	WiggleRoom number,
	HistoryLen number,
	SampleLen number,
	PredictionLen number
);
alter table DataParms add constraint DataParms_PK primary key( ProcessId, DatasetId) using index tablespace LogIdx;

drop table EngineParms purge;
create table EngineParms(
	ProcessId number,
	EngineType number,
	InputCount number,
	OutputCount number,
	WNN_DecompLevel number,
	WNN_WaveletType varchar2(16),
	AdderCount number
);
alter table EngineParms add constraint EngineParms_PK primary key( ProcessId ) using index tablespace LogIdx;

drop table EngineThreads purge;
create table EngineThreads(
	ProcessId number,
	TestId number,
	LayerId number,
	CoreId number,
	CoreType number,
	DatasetId number,
	ThreadId number,
	BasePid number,
	BaseTid number,
	AdderId number
) storage (freelists 8);
alter table EngineThreads add constraint EngineThreads_PK primary key (ProcessId, TestId, LayerId, CoreId, DatasetId) using index tablespace LogIdx;
alter table EngineThreads add constraint EngineThreads_UK1 unique (ProcessId, ThreadId) using index tablespace LogIdx;
--alter table EngineThreads add constraint EngineThreads_UK2 unique (ProcessId, LayerId, CoreId) using index tablespace LogIdx;

drop table CoreImage_SOM purge;
create table CoreImage_SOM(
	ProcessId number,
	ThreadId number,
	FromNeuron number,
	toNeuron number, 
	Weight number
) storage (freelists 8);
alter table CoreImage_SOM add constraint CoreImage_SOM_PK primary key( ProcessId, ThreadId, FromNeuron, ToNeuron ) using index tablespace LogIdx;

drop table CoreImage_SVM purge;
create table CoreImage_SVM(
	ProcessId number,
	ThreadId number,
	SVId number,
	VarId number,
	Weight number
) storage (freelists 8);
alter table CoreImage_SVM add constraint CoreImage_SVM_PK primary key( ProcessId, ThreadId, SVId, VarId ) using index tablespace LogIdx;

drop table CoreParms_NN purge;
create table CoreParms_NN(
	AdderId number,
	ProcessId number,
	LayerId number,
	CoreId number,
	InputCount number,
	OutputCount number,
	LevelsCount number,
	LevelRatioS varchar2(60),
	MaxEpochs number,
	TargetMSE number,
	UseContext number,
	BP_Algo number, 
	TrainingBatchCount number,
	StopAtDivergence number,
	LearningRate number,
	LearningMomentum number,
	ActivationFunction number,
	HCPbeta number,
	Mu number,
	d0 number,
	SCGDmaxK number
);
alter table CoreParms_NN add constraint CoreParms_NN_PK primary key( AdderId, ProcessId, LayerId, CoreId ) using index tablespace LogIdx;

drop table CoreParms_GA purge;
create table CoreParms_GA(
	AdderId number,
	ProcessId number,
	LayerId number,
	CoreId number,
	InputCount number,
	OutputCount number,
	FitnessSizeThreshold number, 
	PopulationSize number, 
	MaxGenerations number,
	Levels number, 
	FitnessSkewingFactor number, 
	TargetFitness number, 
	CrossOverProbability number, 
	MutationProbability number, 
	CrossSelfRate number, 
	Roulette_Max_Tries number, 
	ADF_Force_Presence number, 
	ADF_Tree_FixedValues_Ratio number, 
	ADF_Tree_DataPoints_Ratio number, 
	ADF_Leaf_FixedValues_Ratio number
);
alter table CoreParms_GA add  constraint CoreParms_GA_PK primary key( AdderId, ProcessId, LayerId, CoreId ) using index tablespace LogIdx;

drop table CoreParms_SOM purge;
create table CoreParms_SOM(
	AdderId number,
	ProcessId number,
	LayerId number,
	CoreId number,
	InputCount number,
	OutputCount number,
	MaxEpochs number, 
	TDFunction number, 
	BaseTD number, 
	LRFunction number, 
	BaseLR number
);
alter table CoreParms_SOM add  constraint CoreParms_SOM_PK primary key( AdderId, ProcessId, LayerId, CoreId ) using index tablespace LogIdx;

drop table CoreParms_SVM purge;
create table CoreParms_SVM(
	AdderId number,
	ProcessId number,
	LayerId number,
	CoreId number,
	InputCount number,
	MaxEpochs number, 
	C number,
	Epsilon number,
	IterToShrink number,
	KernelType number,
	PolyDegree number,
	RBFGamma number,
	CoefLin number,
	CoefConst number,
	KernelCacheSize number
);
alter table CoreParms_SVM add  constraint CoreParms_SVM_PK primary key( AdderId, ProcessId, LayerId, CoreId ) using index tablespace LogIdx;

--alter table MSELog add constraint MSELog_AdderId_NN check(AdderId is not null);

drop table CoreLogs_GA purge;
create table CoreLogs_GA(
	ProcessId number,
	ThreadId number,
	ActualGenerations number
) storage (freelists 8);
alter table CoreLogs_GA add constraint CoreLogs_GA_PK primary key(ProcessId, ThreadId) using index tablespace LogIdx;

drop table CoreLogs_NN purge;
create table CoreLogs_NN(
	ProcessId number,
	ThreadId number,
	ActualEpochs number
) storage (initial 100m next 100m freelists 8);
alter table CoreLogs_NN add constraint CoreLogs_NN_PK primary key(ProcessId, ThreadId) using index tablespace LogIdx;

drop   table CoreLogs_NN_SCGD purge;
create table CoreLogs_NN_SCGD(
	ProcessId number,
	ThreadId number,
	Epoch number,
	SampleId number,
	BPid number,
	K number,
	delta number,
	mu number,
	alpha number,
	beta number,
	lambda number,
	lambdau number,
	pnorm number,
	rnorm number,
	enorm number,
	dWnorm number,
	comp number
) storage (freelists 8);
alter table CoreLogs_NN_SCGD add constraint CoreLogs_NN_SCGD_PK primary key(ProcessId, ThreadId, Epoch, BPid, K) using index tablespace LogIdx;

drop   table CoreLogs_SVM purge;
create table CoreLogs_SVM(
	ProcessId number,
	ThreadId number,
	SVcount number,
	ActualEpochs number,
	ThresholdB number,
	maxdiff number,
	L1loss number,
	WVnorm number,
	LEVnorm number,
	KEvCount number
) storage (initial 100m next 100m freelists 8);
alter table CoreLogs_SVM add constraint CoreLogs_SVM_PK primary key(ProcessId, ThreadId) using index tablespace LogIdx;

drop   table CoreLogs_SOM purge;
create table CoreLogs_SOM(
	ProcessId number,
	ThreadId number,
	ActualEpochs number
) storage (freelists 8);
alter table CoreLogs_SOM add constraint CoreLogs_SOM_PK primary key(ProcessId, ThreadId) using index tablespace LogIdx;

-------------------------------------------------------------------------------------- Foreign Key Constraints --------------------------------------------------------------------------------------
alter table TradeInfo add constraint TradeInfo_FK_ClientInfo foreign key(ProcessId) references ClientInfo(ProcessId);
alter table Dataparms add constraint DataParms_FK_ClientInfo foreign key(ProcessId) references ClientInfo(ProcessId);
alter table Engineparms add constraint EngineParms_FK_ClientInfo foreign key(ProcessId) references ClientInfo(ProcessId);
alter table EngineThreads add constraint EngineThreads_FK_EngineParms foreign key(BasePid) references EngineParms(ProcessId);
alter table EngineThreads add constraint EngineThreads_FK_DataParms foreign key(ProcessId, DatasetId) references DataParms(ProcessId, DatasetId);
alter table CoreImage_NN add constraint CoreImage_NN_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreImage_SOM add constraint CoreImage_SOM_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreImage_SVM add constraint CoreImage_SVM_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
--alter table CoreParms_NN add constraint CoreParms_NN_FK_EngineThreads foreign key(ProcessId, LayerId, CoreId) references EngineThreads(ProcessId, LayerId, CoreId);
--alter table CoreParms_GA add constraint CoreParms_GA_FK_EngineThreads foreign key(ProcessId, LayerId, CoreId) references EngineThreads(ProcessId, LayerId, CoreId);
--alter table CoreParms_SOM add constraint CoreParms_SOM_FK_EngineThreads foreign key(ProcessId, LayerId, CoreId) references EngineThreads(ProcessId, LayerId, CoreId);
--alter table CoreParms_SVM add constraint CoreParms_SVM_FK_EngineThreads foreign key(ProcessId, LayerId, CoreId) references EngineThreads(ProcessId, LayerId, CoreId);
alter table MSELog add constraint MSELog_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table RunLog add constraint RunLog_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreLogs_GA add constraint CoreLogs_GA_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreLogs_NN add constraint CoreLogs_NN_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreLogs_NN_SCGD add constraint CoreLogs_NN_SCGD_FK_ET foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreLogs_SVM add constraint CoreLogs_SVM_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
alter table CoreLogs_SOM add constraint CoreLogs_SOM_FK_EngineThreads foreign key(ProcessId, ThreadId) references EngineThreads(ProcessId, ThreadId);
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
