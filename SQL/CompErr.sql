select mse_t from mylog_mse where epoch=999 and processid=&&1;
select avg(errortrs*errortrs) from mylog_run where processid=&&1;
