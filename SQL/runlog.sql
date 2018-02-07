select processid,threadid,pos,featureid,actualtrs,predictedtrs,errortrs from runlog where processid=&&1 order by 1,2,3,4
/
