drop table ClientInfo purge;
create table ClientInfo(
	ProcessId number,
	ClientName varchar2(60),
	ClientStart date,
	Duration number,
	SimulationLen number,
	SimulationStart date,
	DoTraining number,
	DoRun number
);
alter table ClientInfo add constraint ClientInfo_PK primary key( ProcessId ) using index tablespace LogIdx;

drop table TrainLog purge;
create table TrainLog(
	ProcessId number,
	ThreadId number,
	Epoch number,
	MSE_T number,
	MSE_V number
) storage (initial 100m next 100m );
alter table TrainLog add constraint TrainLog_PK primary key( ProcessId, ThreadId, Epoch ) using index tablespace LogIdx;

drop table RunLog purge;
create table RunLog(
	ProcessId number,
	ThreadId number,
	NetProcessId number,
	NetThreadId number,
	Pos number,
	FeatureId number,
	Actual number,
	Predicted number,
	Error number,
	ActualTRS number,
	PredictedTRS number,
	ErrorTRS number,
	BarWidth number,
	ErrorP number
) storage (initial 512m minextents 8 pctincrease 0);
alter table RunLog add constraint RunLog_PK primary key( ProcessId, ThreadId, Pos, FeatureId ) using index tablespace LogIdx;

drop table CoreImage_NN purge;
create table CoreImage_NN(
	ProcessId number,
	ThreadId number,
	Epoch number,
	WId number,
	W number
);
alter table CoreImage_NN add constraint CoreImage_NN_PK primary key( ProcessId, ThreadId, Epoch, WId ) using index tablespace LogIdx;

