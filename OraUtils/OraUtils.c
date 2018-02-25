
/* Result Sets Interface */
#ifndef SQL_CRSR
#  define SQL_CRSR
  struct sql_cursor
  {
    unsigned int curocn;
    void *ptr1;
    void *ptr2;
    unsigned int magic;
  };
  typedef struct sql_cursor sql_cursor;
  typedef struct sql_cursor SQL_CURSOR;
#endif /* SQL_CRSR */

/* Thread Safety */
typedef void * sql_context;
typedef void * SQL_CONTEXT;

/* Object support */
struct sqltvn
{
  unsigned char *tvnvsn; 
  unsigned short tvnvsnl; 
  unsigned char *tvnnm;
  unsigned short tvnnml; 
  unsigned char *tvnsnm;
  unsigned short tvnsnml;
};
typedef struct sqltvn sqltvn;

struct sqladts
{
  unsigned int adtvsn; 
  unsigned short adtmode; 
  unsigned short adtnum;  
  sqltvn adttvn[1];       
};
typedef struct sqladts sqladts;

static struct sqladts sqladt = {
  1,1,0,
};

/* Binding to PL/SQL Records */
struct sqltdss
{
  unsigned int tdsvsn; 
  unsigned short tdsnum; 
  unsigned char *tdsval[1]; 
};
typedef struct sqltdss sqltdss;
static struct sqltdss sqltds =
{
  1,
  0,
};

/* File name & Package Name */
struct sqlcxp
{
  unsigned short fillen;
           char  filnam[12];
};
static const struct sqlcxp sqlfpn =
{
    11,
    "OraUtils.pc"
};


static unsigned int sqlctx = 160403;


static struct sqlexd {
   unsigned int   sqlvsn;
   unsigned int   arrsiz;
   unsigned int   iters;
   unsigned int   offset;
   unsigned short selerr;
   unsigned short sqlety;
   unsigned int   occurs;
      const short *cud;
   unsigned char  *sqlest;
      const char  *stmt;
   sqladts *sqladtp;
   sqltdss *sqltdsp;
            void  **sqphsv;
   unsigned int   *sqphsl;
            int   *sqphss;
            void  **sqpind;
            int   *sqpins;
   unsigned int   *sqparm;
   unsigned int   **sqparc;
   unsigned short  *sqpadto;
   unsigned short  *sqptdso;
   unsigned int   sqlcmax;
   unsigned int   sqlcmin;
   unsigned int   sqlcincr;
   unsigned int   sqlctimeout;
   unsigned int   sqlcnowait;
              int   sqfoff;
   unsigned int   sqcmod;
   unsigned int   sqfmod;
   unsigned int   sqlpfmem;
            void  *sqhstv[6];
   unsigned int   sqhstl[6];
            int   sqhsts[6];
            void  *sqindv[6];
            int   sqinds[6];
   unsigned int   sqharm[6];
   unsigned int   *sqharc[6];
   unsigned short  sqadto[6];
   unsigned short  sqtdso[6];
} sqlstm = {13,6};

/* SQLLIB Prototypes */
extern void sqlcxt (void **, unsigned int *,
                    struct sqlexd *, const struct sqlcxp *);
extern void sqlcx2t(void **, unsigned int *,
                    struct sqlexd *, const struct sqlcxp *);
extern void sqlbuft(void **, char *);
extern void sqlgs2t(void **, char *);
extern void sqlorat(void **, unsigned int *, void *);

/* Forms Interface */
static const int IAPSUCC = 0;
static const int IAPFAIL = 1403;
static const int IAPFTL  = 535;
extern void sqliem(unsigned char *, signed int *);

typedef struct { unsigned short len; unsigned char arr[1]; } VARCHAR;
typedef struct { unsigned short len; unsigned char arr[1]; } varchar;

/* cud (compilation unit data) array */
static const short sqlcud0[] =
{13,4138,178,8,0,
5,0,0,0,0,0,58,102,0,0,1,1,0,1,0,3,109,0,0,
24,0,0,0,0,0,27,104,0,0,4,4,0,1,0,1,97,0,0,1,97,0,0,1,97,0,0,1,10,0,0,
55,0,0,2,0,0,32,127,0,0,0,0,0,1,0,
70,0,0,3,0,0,30,129,0,0,0,0,0,1,0,
85,0,0,4,0,0,29,138,0,0,0,0,0,1,0,
100,0,0,5,0,0,17,155,0,0,1,1,0,1,0,1,97,0,0,
119,0,0,5,0,0,45,157,0,0,0,0,0,1,0,
134,0,0,5,0,0,13,159,0,0,6,0,0,1,0,2,97,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,
4,0,0,
173,0,0,5,0,0,13,176,0,0,6,0,0,1,0,2,97,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,
4,0,0,
212,0,0,5,0,0,15,189,0,0,0,0,0,1,0,
227,0,0,0,0,0,90,248,0,64,0,0,5,105,110,77,83,69,1,0,
247,0,0,0,0,0,93,249,0,64,0,0,5,105,110,77,83,69,1,0,
267,0,0,0,0,0,93,250,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
291,0,0,0,0,0,93,251,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
315,0,0,0,0,0,93,252,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
339,0,0,0,0,0,93,253,0,64,1,1,5,105,110,77,83,69,1,0,3329,4,0,0,
363,0,0,0,0,0,93,254,0,64,1,1,5,105,110,77,83,69,1,0,3329,4,0,0,
387,0,0,6,0,0,17,255,0,0,1,1,0,1,0,1,97,0,0,
406,0,0,6,0,0,23,256,0,64,0,0,5,105,110,77,83,69,1,0,
426,0,0,0,0,0,91,265,0,64,0,0,5,105,110,77,83,69,1,0,
446,0,0,0,0,0,90,329,0,64,0,0,6,105,110,67,73,78,78,1,0,
467,0,0,0,0,0,93,330,0,64,0,0,6,105,110,67,73,78,78,1,0,
488,0,0,0,0,0,93,331,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
513,0,0,0,0,0,93,332,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
538,0,0,0,0,0,93,333,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
563,0,0,0,0,0,93,334,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
588,0,0,0,0,0,93,335,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,4,0,0,
613,0,0,6,0,0,17,336,0,0,1,1,0,1,0,1,97,0,0,
632,0,0,6,0,0,23,337,0,64,0,0,6,105,110,67,73,78,78,1,0,
653,0,0,0,0,0,91,345,0,64,0,0,6,105,110,67,73,78,78,1,0,
674,0,0,7,0,0,24,374,0,0,1,1,0,1,0,1,97,0,0,
693,0,0,0,0,0,90,447,0,64,0,0,5,105,110,82,117,110,1,0,
713,0,0,0,0,0,93,448,0,64,0,0,5,105,110,82,117,110,1,0,
733,0,0,0,0,0,93,449,0,64,1,1,5,105,110,82,117,110,1,0,3329,3,0,0,
757,0,0,0,0,0,93,450,0,64,1,1,5,105,110,82,117,110,1,0,3329,3,0,0,
781,0,0,0,0,0,93,451,0,64,1,1,5,105,110,82,117,110,1,0,3329,3,0,0,
805,0,0,0,0,0,93,452,0,64,1,1,5,105,110,82,117,110,1,0,3329,3,0,0,
829,0,0,0,0,0,93,453,0,64,1,1,5,105,110,82,117,110,1,0,3329,4,0,0,
853,0,0,0,0,0,93,454,0,64,1,1,5,105,110,82,117,110,1,0,3329,4,0,0,
877,0,0,0,0,0,93,455,0,64,1,1,5,105,110,82,117,110,1,0,3329,4,0,0,
901,0,0,6,0,0,17,456,0,0,1,1,0,1,0,1,97,0,0,
920,0,0,6,0,0,23,457,0,64,0,0,5,105,110,82,117,110,1,0,
940,0,0,0,0,0,91,466,0,64,0,0,5,105,110,82,117,110,1,0,
960,0,0,8,0,0,17,500,0,0,1,1,0,1,0,1,97,0,0,
979,0,0,8,0,0,45,502,0,0,0,0,0,1,0,
994,0,0,8,0,0,13,504,0,0,2,0,0,1,0,2,3,0,0,2,4,0,0,
1017,0,0,8,0,0,15,515,0,0,0,0,0,1,0,
};


#include "OraUtils.h"

#include <sqlca.h>


//-- local copies
void Trim(char* str) {
	int l = 0;
	int i;
	int r = (int)strlen(str);
	char ret[MAX_PATH];
	while (isspace(str[l])>0) l++;
	while (isspace(str[r-1])>0) r--;
	for (i = 0; i<(r-l); i++) ret[i] = str[l+i];
	ret[r-l] = '\0';
	strcpy(str, ret);
}
void getCurrentTime(char* ot) {
	time_t mytime = time(NULL);
	sprintf(ot, "%s", ctime(&mytime));
}
void LogWrite(tDbg* dbg, int LogType, const char* msg, int argcount, ...) {
	// pLogLevel=	0 (No screen, No file) | 1 (Screen, No file) | 2 (Screen AND File)
	int n;
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	va_list			arguments;
	char submsg[MAX_PATH];
	unsigned int	im = 0;
	int				prev_im = 0;
	//char timestamp[60];

	if (dbg->level==0&&LogType==DBG_LEVEL_STD) return;

	if (dbg->ThreadSafeLogging>0) WaitForSingleObject(dbg->Mtx, INFINITE);
/*
	//-- Opens Log file only once
	if (dbg->fIsOpen!=1) {
		strcpy(dbg->FullfName, dbg->fPath); strcat(dbg->FullfName, "/"); strcat(dbg->FullfName, dbg->fName);
		dbg->outFile->handle = fopen(dbg->FullfName, "a");
		dbg->fIsOpen = 1;
		getCurrentTime(timestamp);
		fprintf(dbg->outFile->handle, "\n---------- Process %d Started New Log at %s ----------\n", GetCurrentProcessId(), timestamp);
	}
*/
	va_start(arguments, argcount);
	n = 0;

	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			n++;
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				if (dbg->level==1||dbg->level==3||LogType==DBG_LEVEL_ERR) printf(submsg, arg_s);
				if (dbg->level==2||dbg->level==3||LogType==DBG_LEVEL_ERR)	fprintf(dbg->outFile->handle, submsg, arg_s);
			} else if (msg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				if (dbg->level==1||dbg->level==3||LogType==DBG_LEVEL_ERR) printf(submsg, arg_d);
				if (dbg->level==2||dbg->level==3||LogType==DBG_LEVEL_ERR)	fprintf(dbg->outFile->handle, submsg, arg_d);
			} else if (msg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				if (dbg->level==1||dbg->level==3||LogType==DBG_LEVEL_ERR) printf(submsg, arg_d);
				if (dbg->level==2||dbg->level==3||LogType==DBG_LEVEL_ERR)	fprintf(dbg->outFile->handle, submsg, arg_d);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				if (dbg->level==1||dbg->level==3||LogType==DBG_LEVEL_ERR) printf(submsg, arg_f);
				if (dbg->level==2||dbg->level==3||LogType==DBG_LEVEL_ERR)	fprintf(dbg->outFile->handle, submsg, arg_f);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	if (dbg->level==1||dbg->level==3||LogType==DBG_LEVEL_ERR) printf(submsg);
	if (dbg->level==2||dbg->level==3||LogType==DBG_LEVEL_ERR) fprintf(dbg->outFile->handle, submsg);
	if (LogType==DBG_LEVEL_ERR && dbg->PauseOnError) { printf("Press any key..."); getchar(); }

	va_end(arguments);

	if (dbg->ThreadSafeLogging>0) ReleaseMutex(dbg->Mtx);
}

//=== DB common functions
EXPORT bool  OraConnect(tDbg* dbg, tDBConnection* DBConnInfo) {

	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	char* username = DBConnInfo->DBUser;
	char* password = DBConnInfo->DBPassword;
	char* dbstring = DBConnInfo->DBConnString;
	sql_context vCtx = NULL;
	/* EXEC SQL END DECLARE SECTION; */ 


	//LogWrite_C(dbg, "%s About to execute connect with %s , %s , %s\n", 4, timestamp_C, pUserName, pPassword, pDBString);

	char* vPath = getenv("PATH");
	char* vOH = getenv("ORACLE_HOME");
	//printf("PATH=%s\nORACLE_HOME=%s\n", vPath, vOH);

	/* EXEC SQL CONTEXT ALLOCATE : vCtx; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 1;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )5;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)&vCtx;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(void *);
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL CONNECT : username IDENTIFIED BY : password USING : dbstring; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 4;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )10;
 sqlstm.offset = (unsigned int  )24;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)username;
 sqlstm.sqhstl[0] = (unsigned int  )0;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqhstv[1] = (         void  *)password;
 sqlstm.sqhstl[1] = (unsigned int  )0;
 sqlstm.sqhsts[1] = (         int  )0;
 sqlstm.sqindv[1] = (         void  *)0;
 sqlstm.sqinds[1] = (         int  )0;
 sqlstm.sqharm[1] = (unsigned int  )0;
 sqlstm.sqadto[1] = (unsigned short )0;
 sqlstm.sqtdso[1] = (unsigned short )0;
 sqlstm.sqhstv[2] = (         void  *)dbstring;
 sqlstm.sqhstl[2] = (unsigned int  )0;
 sqlstm.sqhsts[2] = (         int  )0;
 sqlstm.sqindv[2] = (         void  *)0;
 sqlstm.sqinds[2] = (         int  )0;
 sqlstm.sqharm[2] = (unsigned int  )0;
 sqlstm.sqadto[2] = (unsigned short )0;
 sqlstm.sqtdso[2] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlstm.sqlcmax = (unsigned int )100;
 sqlstm.sqlcmin = (unsigned int )2;
 sqlstm.sqlcincr = (unsigned int )1;
 sqlstm.sqlctimeout = (unsigned int )0;
 sqlstm.sqlcnowait = (unsigned int )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	if (sqlca.sqlcode==0) {
		//memcpy(DBConnInfo->Ctx, vCtx, sizeof(sql_context));
		DBConnInfo->DBCtx = vCtx;
		//dbg->DBCtx = vCtx;
		LogWrite(dbg, DBG_LEVEL_STD, "OraConnect() - Connected to ORACLE as user: %s ; DBConnInfo->DBCtx=%p\n", 2, username, DBConnInfo->DBCtx);
	} else {
		LogWrite(dbg, DBG_LEVEL_STD, "PATH=%s\n", 1, vPath);
		LogWrite(dbg, DBG_LEVEL_STD, "ORACLE_HOME=%s\n", 1, vOH);
		LogWrite(dbg, DBG_LEVEL_STD, "DBUser=%s\n", 1, DBConnInfo->DBUser);
		//Oerr(dbg, __func__, sqlca.sqlcode);
		LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. user = %s\n Error %s", 3, __func__, DBConnInfo->DBUser, sqlca.sqlerrm.sqlerrmc);
	}
	return(sqlca.sqlcode);
}
EXPORT void OraDisconnect(tDBConnection* DBConnInfo, int Commit) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = DBConnInfo->DBCtx;
	/* EXEC SQL END   DECLARE SECTION; */ 


	/* EXEC SQL CONTEXT USE : vCtx; */ 

	if (Commit==0) {
		/* EXEC SQL ROLLBACK RELEASE; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 4;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )55;
  sqlstm.cud = sqlcud0;
  sqlstm.sqlest = (unsigned char  *)&sqlca;
  sqlstm.sqlety = (unsigned short)4352;
  sqlstm.occurs = (unsigned int  )0;
  sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	} else {
		/* EXEC SQL COMMIT WORK RELEASE; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 4;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )70;
  sqlstm.cud = sqlcud0;
  sqlstm.sqlest = (unsigned char  *)&sqlca;
  sqlstm.sqlety = (unsigned short)4352;
  sqlstm.occurs = (unsigned int  )0;
  sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	}
}
EXPORT void OraCommit(tDBConnection* DBConnInfo) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = DBConnInfo->DBCtx;
	/* EXEC SQL END   DECLARE SECTION; */ 


	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL COMMIT; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 4;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )85;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


}

//=== Retrieval functions
EXPORT bool Ora_GetFlatOHLCV(tDbg* dbg, tDBConnection* db, char* pSymbol, char* pTF, char* pDate0, int pRecCount, char** oBarTime, float* oBarData, char* oBaseTime, float* oBaseBar) {
	//-- 'Flat' bar means returning OHLC(V) values for each record as an array 
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = db->DBCtx;
	int retval = 0;
	int i;
	char stmt[1000];
	char vNDT[12+1];
	float vOpen, vHigh, vLow, vClose, vVolume;
	/* EXEC SQL END   DECLARE SECTION; */ 

	sprintf(stmt, "select to_char(newdatetime,'YYYYMMDDHH24MI'), open, high, low, close, nvl(volume,0) from %s_%s where NewDateTime<=to_date('%s','YYYYMMDDHH24MI') order by 1 desc", pSymbol, pTF, pDate0);

	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL PREPARE sOHLCV FROM : stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 4;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )100;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL DECLARE cOHLCV CURSOR FOR sOHLCV; */ 

	/* EXEC SQL OPEN cOHLCV; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 4;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )119;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqcmod = (unsigned int )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	for (i = pRecCount-1; i>=0; i--) {
		/* EXEC SQL FETCH cOHLCV into : vNDT, : vOpen, : vHigh, : vLow, : vClose, : vVolume; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 6;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )134;
  sqlstm.selerr = (unsigned short)1;
  sqlstm.sqlpfmem = (unsigned int  )0;
  sqlstm.cud = sqlcud0;
  sqlstm.sqlest = (unsigned char  *)&sqlca;
  sqlstm.sqlety = (unsigned short)4352;
  sqlstm.occurs = (unsigned int  )0;
  sqlstm.sqfoff = (           int )0;
  sqlstm.sqfmod = (unsigned int )2;
  sqlstm.sqhstv[0] = (         void  *)vNDT;
  sqlstm.sqhstl[0] = (unsigned int  )13;
  sqlstm.sqhsts[0] = (         int  )0;
  sqlstm.sqindv[0] = (         void  *)0;
  sqlstm.sqinds[0] = (         int  )0;
  sqlstm.sqharm[0] = (unsigned int  )0;
  sqlstm.sqadto[0] = (unsigned short )0;
  sqlstm.sqtdso[0] = (unsigned short )0;
  sqlstm.sqhstv[1] = (         void  *)&vOpen;
  sqlstm.sqhstl[1] = (unsigned int  )sizeof(float);
  sqlstm.sqhsts[1] = (         int  )0;
  sqlstm.sqindv[1] = (         void  *)0;
  sqlstm.sqinds[1] = (         int  )0;
  sqlstm.sqharm[1] = (unsigned int  )0;
  sqlstm.sqadto[1] = (unsigned short )0;
  sqlstm.sqtdso[1] = (unsigned short )0;
  sqlstm.sqhstv[2] = (         void  *)&vHigh;
  sqlstm.sqhstl[2] = (unsigned int  )sizeof(float);
  sqlstm.sqhsts[2] = (         int  )0;
  sqlstm.sqindv[2] = (         void  *)0;
  sqlstm.sqinds[2] = (         int  )0;
  sqlstm.sqharm[2] = (unsigned int  )0;
  sqlstm.sqadto[2] = (unsigned short )0;
  sqlstm.sqtdso[2] = (unsigned short )0;
  sqlstm.sqhstv[3] = (         void  *)&vLow;
  sqlstm.sqhstl[3] = (unsigned int  )sizeof(float);
  sqlstm.sqhsts[3] = (         int  )0;
  sqlstm.sqindv[3] = (         void  *)0;
  sqlstm.sqinds[3] = (         int  )0;
  sqlstm.sqharm[3] = (unsigned int  )0;
  sqlstm.sqadto[3] = (unsigned short )0;
  sqlstm.sqtdso[3] = (unsigned short )0;
  sqlstm.sqhstv[4] = (         void  *)&vClose;
  sqlstm.sqhstl[4] = (unsigned int  )sizeof(float);
  sqlstm.sqhsts[4] = (         int  )0;
  sqlstm.sqindv[4] = (         void  *)0;
  sqlstm.sqinds[4] = (         int  )0;
  sqlstm.sqharm[4] = (unsigned int  )0;
  sqlstm.sqadto[4] = (unsigned short )0;
  sqlstm.sqtdso[4] = (unsigned short )0;
  sqlstm.sqhstv[5] = (         void  *)&vVolume;
  sqlstm.sqhstl[5] = (unsigned int  )sizeof(float);
  sqlstm.sqhsts[5] = (         int  )0;
  sqlstm.sqindv[5] = (         void  *)0;
  sqlstm.sqinds[5] = (         int  )0;
  sqlstm.sqharm[5] = (unsigned int  )0;
  sqlstm.sqadto[5] = (unsigned short )0;
  sqlstm.sqtdso[5] = (unsigned short )0;
  sqlstm.sqphsv = sqlstm.sqhstv;
  sqlstm.sqphsl = sqlstm.sqhstl;
  sqlstm.sqphss = sqlstm.sqhsts;
  sqlstm.sqpind = sqlstm.sqindv;
  sqlstm.sqpins = sqlstm.sqinds;
  sqlstm.sqparm = sqlstm.sqharm;
  sqlstm.sqparc = sqlstm.sqharc;
  sqlstm.sqpadto = sqlstm.sqadto;
  sqlstm.sqptdso = sqlstm.sqtdso;
  sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


		if (sqlca.sqlcode==0) {
			strcpy(oBarTime[i], vNDT);
			oBarData[5*i+0] = vOpen;
			oBarData[5*i+1] = vHigh;
			oBarData[5*i+2] = vLow;
			oBarData[5*i+3] = vClose;
			oBarData[5*i+4] = vVolume;
		} else if (sqlca.sqlcode==1403) {
			break;
		} else {
			LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
			retval = sqlca.sqlcode;
			break;
		}
	}
	//-- one more fetch to get baseBar
	/* EXEC SQL FETCH cOHLCV into : vNDT, : vOpen, : vHigh, : vLow, : vClose, : vVolume; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )173;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqfoff = (           int )0;
 sqlstm.sqfmod = (unsigned int )2;
 sqlstm.sqhstv[0] = (         void  *)vNDT;
 sqlstm.sqhstl[0] = (unsigned int  )13;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqhstv[1] = (         void  *)&vOpen;
 sqlstm.sqhstl[1] = (unsigned int  )sizeof(float);
 sqlstm.sqhsts[1] = (         int  )0;
 sqlstm.sqindv[1] = (         void  *)0;
 sqlstm.sqinds[1] = (         int  )0;
 sqlstm.sqharm[1] = (unsigned int  )0;
 sqlstm.sqadto[1] = (unsigned short )0;
 sqlstm.sqtdso[1] = (unsigned short )0;
 sqlstm.sqhstv[2] = (         void  *)&vHigh;
 sqlstm.sqhstl[2] = (unsigned int  )sizeof(float);
 sqlstm.sqhsts[2] = (         int  )0;
 sqlstm.sqindv[2] = (         void  *)0;
 sqlstm.sqinds[2] = (         int  )0;
 sqlstm.sqharm[2] = (unsigned int  )0;
 sqlstm.sqadto[2] = (unsigned short )0;
 sqlstm.sqtdso[2] = (unsigned short )0;
 sqlstm.sqhstv[3] = (         void  *)&vLow;
 sqlstm.sqhstl[3] = (unsigned int  )sizeof(float);
 sqlstm.sqhsts[3] = (         int  )0;
 sqlstm.sqindv[3] = (         void  *)0;
 sqlstm.sqinds[3] = (         int  )0;
 sqlstm.sqharm[3] = (unsigned int  )0;
 sqlstm.sqadto[3] = (unsigned short )0;
 sqlstm.sqtdso[3] = (unsigned short )0;
 sqlstm.sqhstv[4] = (         void  *)&vClose;
 sqlstm.sqhstl[4] = (unsigned int  )sizeof(float);
 sqlstm.sqhsts[4] = (         int  )0;
 sqlstm.sqindv[4] = (         void  *)0;
 sqlstm.sqinds[4] = (         int  )0;
 sqlstm.sqharm[4] = (unsigned int  )0;
 sqlstm.sqadto[4] = (unsigned short )0;
 sqlstm.sqtdso[4] = (unsigned short )0;
 sqlstm.sqhstv[5] = (         void  *)&vVolume;
 sqlstm.sqhstl[5] = (unsigned int  )sizeof(float);
 sqlstm.sqhsts[5] = (         int  )0;
 sqlstm.sqindv[5] = (         void  *)0;
 sqlstm.sqinds[5] = (         int  )0;
 sqlstm.sqharm[5] = (unsigned int  )0;
 sqlstm.sqadto[5] = (unsigned short )0;
 sqlstm.sqtdso[5] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	if (sqlca.sqlcode==0) {
		strcpy(oBaseTime, vNDT);
		oBaseBar[0] = vOpen;
		oBaseBar[1] = vHigh;
		oBaseBar[2] = vLow;
		oBaseBar[3] = vClose;
		oBaseBar[4] = vVolume;
	} else if (sqlca.sqlcode!=1403) {
		LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		retval = sqlca.sqlcode;
	}

	/* EXEC SQL CLOSE cOHLCV; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )212;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	return retval;
}

//=== Logging functions
EXPORT bool Ora_LogSaveMSE(tDbg* dbg, tDBConnection* db, int pid, int tid, int mseCnt, float* mseT, float* mseV) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int vInsertCount;
	sql_context vCtx = db->DBCtx;
	char stmt[1000];
	//--
	int* vProcessId;
	int* vThreadId;
	int* vEpoch;
	double* vMSE_T;
	double* vMSE_V;
	//--
	int dbl_type = 22;	// Oracle SQLT_BDOUBLE
	int int_type = 3;	// 
	unsigned int dbl_len = sizeof(double);
	unsigned int vchar_type = 96;
	unsigned int vchar_len = 12+1;

	/* EXEC SQL END   DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(dbg, db)!=0) {
			LogWrite(dbg, DBG_LEVEL_ERR, "%s() could not connect to Log Database...\n", 1, __func__);
			return -1;
		}
		vCtx = db->DBCtx;
	}
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 2 - LogDB->DBCtx=%p , vCtx=%p\n", 3, __func__, db->DBCtx, vCtx);

	vInsertCount=mseCnt;
	vProcessId=(int*)malloc(vInsertCount*sizeof(int));
	vThreadId=(int*)malloc(vInsertCount*sizeof(int));
	vEpoch = (int*)malloc(vInsertCount*sizeof(int));
	vMSE_T = (double*)malloc(vInsertCount*sizeof(double));
	vMSE_V = (double*)malloc(vInsertCount*sizeof(double));

	for (i = 0; i < vInsertCount; i++) {
		vProcessId[i]=pid;
		vThreadId[i]=tid;
		vEpoch[i] = i;
		vMSE_T[i] = mseT[i];
		vMSE_V[i] = mseV[i];
	}

	//-- Then, Build the Insert statement
	sprintf(&stmt[0], "insert into TrainLog(ProcessId, ThreadId, Epoch, MSE_T, MSE_V) values(:P01, :P02, :P03, :P04, :P05)");
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 4 - stmt='%s' ; pid[0]=%d ; tid[0]=%d\n", 4, __func__, stmt, pid, tid);

	/* EXEC SQL CONTEXT USE :vCtx; */ 

	//EXEC SQL ALTER SESSION SET EVENTS '10046 trace name context forever, level 4';
	//EXEC SQL ALTER SESSION SET SQL_TRACE = TRUE;
	/* EXEC SQL FOR :vInsertCount ALLOCATE DESCRIPTOR 'inMSE'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )227;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )100;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL SET DESCRIPTOR 'inMSE' COUNT = 5; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )247;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )5;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inMSE' VALUE 1 DATA = :vProcessId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )267;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )1;
 sqlstm.sqhstv[0] = (         void  *)vProcessId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inMSE' VALUE 2 DATA = :vThreadId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )291;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )2;
 sqlstm.sqhstv[0] = (         void  *)vThreadId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inMSE' VALUE 3 DATA = :vEpoch; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )315;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )3;
 sqlstm.sqhstv[0] = (         void  *)vEpoch;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inMSE' VALUE 4 DATA = :vMSE_T; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )339;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )4;
 sqlstm.sqhstv[0] = (         void  *)vMSE_T;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inMSE' VALUE 5 DATA = :vMSE_V; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )363;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )5;
 sqlstm.sqhstv[0] = (         void  *)vMSE_V;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL PREPARE DynIns FROM :stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )387;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount EXECUTE DynIns USING DESCRIPTOR 'inMSE'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )406;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	//EXEC SQL ALTER SESSION SET SQL_TRACE = FALSE;

	//pInsertCount[vNetLevel] = sqlca.sqlerrd[2];
	LogWrite(dbg, DBG_LEVEL_STD, "BulkMSEInsert() inserted %d rows.\n", 1, sqlca.sqlerrd[2]);
	if (sqlca.sqlcode!=0) {
		if (sqlca.sqlcode!=1) LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	/* EXEC SQL DEALLOCATE DESCRIPTOR 'inMSE'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )426;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	//-- free()s
	free(vProcessId);
	free(vThreadId);
	free(vEpoch);
	free(vMSE_T);
	free(vMSE_V);

	return sqlca.sqlcode;
}
EXPORT bool Ora_LogSaveW(tDbg* dbg, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* W) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int vInsertCount;
	sql_context vCtx = db->DBCtx;
	char stmt[1000];
	//--
	int* vProcessId;
	int* vThreadId;
	int* vEpoch;
	int* vWid;
	double* vW;
	//--
	int dbl_type = 22;	// Oracle SQLT_BDOUBLE
	int int_type = 3;	// 
	unsigned int dbl_len = sizeof(double);
	unsigned int vchar_type = 96;
	unsigned int vchar_len = 12+1;

	/* EXEC SQL END   DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(dbg, db)!=0) {
			LogWrite(dbg, DBG_LEVEL_ERR, "%s() could not connect to Log Database...\n", 1, __func__);
			return -1;
		}
		vCtx = db->DBCtx;
	}
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 2 - LogDB->DBCtx=%p , vCtx=%p\n", 3, __func__, db->DBCtx, vCtx);

	vInsertCount=Wcnt;
	vProcessId=(int*)malloc(vInsertCount*sizeof(int));
	vThreadId=(int*)malloc(vInsertCount*sizeof(int));
	vEpoch = (int*)malloc(vInsertCount*sizeof(int));
	vWid = (int*)malloc(vInsertCount*sizeof(int));
	vW = (double*)malloc(vInsertCount*sizeof(double));

	for (i = 0; i < vInsertCount; i++) {
		vProcessId[i]=pid;
		vThreadId[i]=tid;
		vEpoch[i] = epoch;
		vWid[i]=i;
		vW[i]=W[i];
	}

	//-- Then, Build the Insert statement
	sprintf(&stmt[0], "insert into CoreImage_NN (ProcessId, ThreadId, Epoch, WId, W) values(:P01, :P02, :P03, :P04, :P05)");
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 4 - stmt='%s' ; pid[0]=%d ; tid[0]=%d\n", 4, __func__, stmt, pid, tid);

	/* EXEC SQL CONTEXT USE :vCtx; */ 

	//EXEC SQL ALTER SESSION SET EVENTS '10046 trace name context forever, level 4';
	//EXEC SQL ALTER SESSION SET SQL_TRACE = TRUE;
	/* EXEC SQL FOR :vInsertCount ALLOCATE DESCRIPTOR 'inCINN'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )446;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )100;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL SET DESCRIPTOR 'inCINN' COUNT = 5; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )467;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )5;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inCINN' VALUE 1 DATA = :vProcessId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )488;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )1;
 sqlstm.sqhstv[0] = (         void  *)vProcessId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inCINN' VALUE 2 DATA = :vThreadId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )513;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )2;
 sqlstm.sqhstv[0] = (         void  *)vThreadId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inCINN' VALUE 3 DATA = :vEpoch; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )538;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )3;
 sqlstm.sqhstv[0] = (         void  *)vEpoch;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inCINN' VALUE 4 DATA = :vWid; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )563;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )4;
 sqlstm.sqhstv[0] = (         void  *)vWid;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inCINN' VALUE 5 DATA = :vW; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )588;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )5;
 sqlstm.sqhstv[0] = (         void  *)vW;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL PREPARE DynIns FROM :stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )613;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount EXECUTE DynIns USING DESCRIPTOR 'inCINN'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )632;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	//EXEC SQL ALTER SESSION SET SQL_TRACE = FALSE;

	LogWrite(dbg, DBG_LEVEL_STD, "%s() inserted %d rows.\n", 2, __func__, sqlca.sqlerrd[2]);
	if (sqlca.sqlcode!=0) {
		if (sqlca.sqlcode!=1) LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	/* EXEC SQL DEALLOCATE DESCRIPTOR 'inCINN'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )653;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	//-- free()s
	free(vProcessId);
	free(vThreadId);
	free(vEpoch);
	free(vWid);
	free(vW);

	return sqlca.sqlcode;
}
EXPORT bool Ora_LogSaveClient(tDbg* dbg, tDBConnection* db, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTraining, bool doRun) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = db->DBCtx;
	char stmt[1000];
	/* EXEC SQL END DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(dbg, db)!=0) {
			printf("%s() could not connect to Log Database...\n", __func__);
			return -1;
		}
		vCtx = db->DBCtx;
	}
	//-- Builds Insert statement
	sprintf(&stmt[0], "insert into ClientInfo(ProcessId, ClientName, ClientStart, SimulationLen, Duration, SimulationStart, DoTraining, DoRun) values(%d, '%s', sysdate, %d, %ld, to_date('%s','YYYYMMDDHH24MI'), %d, %d)", pid, clientName, simulLen, (DWORD)(duration/1000), simulStart, (doTraining)?1:0, (doRun)?1:0);
	//-- Executes Insert statement
	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL EXECUTE IMMEDIATE : stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )674;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	if (sqlca.sqlcode!=0) {
		LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	return 0;
}
EXPORT bool Ora_LogSaveRun(tDbg* dbg, tDBConnection* db, int pid, int tid, int barCnt, int featuresCnt, numtype* prediction, numtype* actual) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i, b, f;
	int vInsertCount;
	int vFeaturesCnt;
	sql_context vCtx = db->DBCtx;
	char stmt[1000];
	//--
	int* vProcessId;
	int* vThreadId;
	int* vStep;
	int* vFeatureId;
	double* vPredictionTRS;
	double* vActualTRS;
	double* vErrorTRS;
	//--
	int dbl_type = 22;	// Oracle SQLT_BDOUBLE
	int int_type = 3;	// 
	unsigned int dbl_len = sizeof(double);
	unsigned int vchar_type = 96;
	unsigned int vchar_len = 12+1;

	/* EXEC SQL END   DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(dbg, db)!=0) {
			LogWrite(dbg, DBG_LEVEL_ERR, "%s() could not connect to Log Database...\n", 1, __func__);
			return -1;
		}
		vCtx = db->DBCtx;
	}
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 2 - LogDB->DBCtx=%p , vCtx=%p\n", 3, __func__, db->DBCtx, vCtx);

	vFeaturesCnt=featuresCnt;
	vInsertCount=barCnt*featuresCnt;
	vProcessId=(int*)malloc(vInsertCount*sizeof(int));
	vThreadId=(int*)malloc(vInsertCount*sizeof(int));
	vStep = (int*)malloc(vInsertCount*sizeof(int));
	vFeatureId = (int*)malloc(vInsertCount*sizeof(int));
	vPredictionTRS=(double*)malloc(vInsertCount*sizeof(double));
	vActualTRS=(double*)malloc(vInsertCount*sizeof(double));
	vErrorTRS=(double*)malloc(vInsertCount*sizeof(double));

	i=0;
	for (b=0; b<barCnt; b++) {
		for (f=0; f<featuresCnt; f++) {
			vProcessId[i]=pid;
			vThreadId[i]=tid;
			vStep[i] = b;
			vFeatureId[i]=f;
			vPredictionTRS[i]=prediction[i];
			vActualTRS[i]=actual[i];
			vErrorTRS[i]=fabs(actual[i]-prediction[i]);

			i++;
		}
	}

	//-- Then, Build the Insert statement
	sprintf(&stmt[0], "insert into RunLog (ProcessId, ThreadId, Pos, FeatureId, PredictedTRS, ActualTRS, ErrorTRS) values(:P01, :P02, :P03, :P04, :P05, :P06, :P07)");
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 4 - stmt='%s' ; pid[0]=%d ; tid[0]=%d\n", 4, __func__, stmt, pid, tid);

	/* EXEC SQL CONTEXT USE :vCtx; */ 

	//EXEC SQL ALTER SESSION SET EVENTS '10046 trace name context forever, level 4';
	//EXEC SQL ALTER SESSION SET SQL_TRACE = TRUE;
	/* EXEC SQL FOR :vInsertCount ALLOCATE DESCRIPTOR 'inRun'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )693;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )100;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL SET DESCRIPTOR 'inRun' COUNT = 7; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )713;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )7;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inRun' VALUE 1 DATA = :vProcessId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )733;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )1;
 sqlstm.sqhstv[0] = (         void  *)vProcessId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inRun' VALUE 2 DATA = :vThreadId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )757;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )2;
 sqlstm.sqhstv[0] = (         void  *)vThreadId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR : vInsertCount SET DESCRIPTOR 'inRun' VALUE 3 DATA = : vStep; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )781;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )3;
 sqlstm.sqhstv[0] = (         void  *)vStep;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR : vInsertCount SET DESCRIPTOR 'inRun' VALUE 4 DATA = : vFeatureId; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )805;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )4;
 sqlstm.sqhstv[0] = (         void  *)vFeatureId;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
 sqlstm.sqhsts[0] = (         int  )sizeof(int);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inRun' VALUE 5 DATA = :vPredictionTRS; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )829;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )5;
 sqlstm.sqhstv[0] = (         void  *)vPredictionTRS;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inRun' VALUE 6 DATA = :vActualTRS; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )853;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )6;
 sqlstm.sqhstv[0] = (         void  *)vActualTRS;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount SET DESCRIPTOR 'inRun' VALUE  7 DATA = :vErrorTRS; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )877;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )7;
 sqlstm.sqhstv[0] = (         void  *)vErrorTRS;
 sqlstm.sqhstl[0] = (unsigned int  )sizeof(double);
 sqlstm.sqhsts[0] = (         int  )sizeof(double);
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL PREPARE DynIns FROM :stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )901;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL FOR :vInsertCount EXECUTE DynIns USING DESCRIPTOR 'inRun'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )vInsertCount;
 sqlstm.offset = (unsigned int  )920;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	//EXEC SQL ALTER SESSION SET SQL_TRACE = FALSE;

	//pInsertCount[vNetLevel] = sqlca.sqlerrd[2];
	LogWrite(dbg, DBG_LEVEL_STD, "%s() inserted %d rows.\n", 2, __func__, sqlca.sqlerrd[2]);
	if (sqlca.sqlcode!=0) {
		if (sqlca.sqlcode!=1) LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	/* EXEC SQL DEALLOCATE DESCRIPTOR 'inRun'; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )940;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	//-- free()s
	free(vProcessId);
	free(vThreadId);
	free(vStep);
	free(vFeatureId);
	free(vPredictionTRS);
	free(vActualTRS);
	free(vErrorTRS);

	return sqlca.sqlcode;
}

EXPORT bool Ora_LogLoadW(tDbg* dbg, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* oW) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int retval = 0;
	sql_context vCtx = db->DBCtx;
	char stmt[1000];
	//-- Network Weights variables
	int vWid; double vW;
	/* EXEC SQL END DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(dbg, db)!=0) printf("%s() could not connect to Log Database...\n", __func__);
		vCtx = db->DBCtx;
	}

	//-- Load Network Weights from CoreImage_NN
	sprintf(&stmt[0], "select WId, W from CoreImage_NN where ProcessId=%d and ThreadId=%d and Epoch=%d order by 1,2,3", pid, tid, epoch);
	LogWrite(dbg, DBG_LEVEL_STD, "%s() CheckPoint 3 - stmt=%s\n", 2, __func__, stmt);
	/* EXEC SQL CONTEXT USE :vCtx; */ 

	/* EXEC SQL PREPARE sLoadW FROM :stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )960;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqhstv[0] = (         void  *)stmt;
 sqlstm.sqhstl[0] = (unsigned int  )1000;
 sqlstm.sqhsts[0] = (         int  )0;
 sqlstm.sqindv[0] = (         void  *)0;
 sqlstm.sqinds[0] = (         int  )0;
 sqlstm.sqharm[0] = (unsigned int  )0;
 sqlstm.sqadto[0] = (unsigned short )0;
 sqlstm.sqtdso[0] = (unsigned short )0;
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL DECLARE cLoadW CURSOR FOR sLoadW; */ 

	/* EXEC SQL OPEN cLoadW; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )979;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqcmod = (unsigned int )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	for(i=0; i<Wcnt; i++) {
		/* EXEC SQL FETCH cLoadW INTO vWid, vW; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 6;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )994;
  sqlstm.selerr = (unsigned short)1;
  sqlstm.sqlpfmem = (unsigned int  )0;
  sqlstm.cud = sqlcud0;
  sqlstm.sqlest = (unsigned char  *)&sqlca;
  sqlstm.sqlety = (unsigned short)4352;
  sqlstm.occurs = (unsigned int  )0;
  sqlstm.sqfoff = (           int )0;
  sqlstm.sqfmod = (unsigned int )2;
  sqlstm.sqhstv[0] = (         void  *)&vWid;
  sqlstm.sqhstl[0] = (unsigned int  )sizeof(int);
  sqlstm.sqhsts[0] = (         int  )0;
  sqlstm.sqindv[0] = (         void  *)0;
  sqlstm.sqinds[0] = (         int  )0;
  sqlstm.sqharm[0] = (unsigned int  )0;
  sqlstm.sqadto[0] = (unsigned short )0;
  sqlstm.sqtdso[0] = (unsigned short )0;
  sqlstm.sqhstv[1] = (         void  *)&vW;
  sqlstm.sqhstl[1] = (unsigned int  )sizeof(double);
  sqlstm.sqhsts[1] = (         int  )0;
  sqlstm.sqindv[1] = (         void  *)0;
  sqlstm.sqinds[1] = (         int  )0;
  sqlstm.sqharm[1] = (unsigned int  )0;
  sqlstm.sqadto[1] = (unsigned short )0;
  sqlstm.sqtdso[1] = (unsigned short )0;
  sqlstm.sqphsv = sqlstm.sqhstv;
  sqlstm.sqphsl = sqlstm.sqhstl;
  sqlstm.sqphss = sqlstm.sqhsts;
  sqlstm.sqpind = sqlstm.sqindv;
  sqlstm.sqpins = sqlstm.sqinds;
  sqlstm.sqparm = sqlstm.sqharm;
  sqlstm.sqparc = sqlstm.sqharc;
  sqlstm.sqpadto = sqlstm.sqadto;
  sqlstm.sqptdso = sqlstm.sqtdso;
  sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


		if (sqlca.sqlcode==0) {
			oW[i] = (numtype)vW;
		} else if (sqlca.sqlcode==1403) {
			break;
		} else {
			LogWrite(dbg, DBG_LEVEL_ERR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
			retval = sqlca.sqlcode;
			break;
		}
	}
	/* EXEC SQL CLOSE cLoadW; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )1017;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	return retval;
}