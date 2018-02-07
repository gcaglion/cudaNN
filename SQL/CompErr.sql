
select mse_t from trainlog where processid=&&1 and epoch=(
select max(epoch) from trainlog where processid=&&1
);
select sum(errortrs*errortrs) SE, avg(errortrs*errortrs) MSE from runlog where processid=&&1;
select featureid, sum(errortrs*errortrs) SE, avg(errortrs*errortrs) MSE from runlog where processid=&&1 group by featureid order by 1;
