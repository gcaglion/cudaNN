
select mse_t from mylog_mse where processid=&&1 and epoch=(
select max(epoch) from mylog_mse where processid=&&1
);
select avg(errortrs*errortrs) from mylog_run where processid=&&1;
