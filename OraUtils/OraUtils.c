
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
100,0,0,5,0,0,17,154,0,0,1,1,0,1,0,1,97,0,0,
119,0,0,5,0,0,45,156,0,0,0,0,0,1,0,
134,0,0,5,0,0,13,158,0,0,5,0,0,1,0,2,97,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,4,0,0,
169,0,0,5,0,0,13,173,0,0,5,0,0,1,0,2,97,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,4,0,0,
204,0,0,5,0,0,15,184,0,0,0,0,0,1,0,
219,0,0,6,0,0,17,213,0,0,1,1,0,1,0,1,97,0,0,
238,0,0,6,0,0,45,215,0,0,0,0,0,1,0,
253,0,0,6,0,0,13,217,0,0,6,0,0,1,0,2,97,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,4,0,0,2,
4,0,0,
292,0,0,6,0,0,15,228,0,0,0,0,0,1,0,
307,0,0,7,0,0,17,247,0,0,1,1,0,1,0,1,97,0,0,
326,0,0,7,0,0,45,249,0,0,0,0,0,1,0,
341,0,0,7,0,0,13,254,0,0,1,0,0,1,0,2,97,0,0,
360,0,0,7,0,0,15,260,0,0,0,0,0,1,0,
375,0,0,8,0,0,17,272,0,0,1,1,0,1,0,1,97,0,0,
394,0,0,8,0,0,45,274,0,0,0,0,0,1,0,
409,0,0,8,0,0,13,280,0,0,1,0,0,1,0,2,97,0,0,
428,0,0,8,0,0,15,287,0,0,0,0,0,1,0,
443,0,0,0,0,0,90,345,0,64,0,0,5,105,110,77,83,69,1,0,
463,0,0,0,0,0,93,346,0,64,0,0,5,105,110,77,83,69,1,0,
483,0,0,0,0,0,93,347,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
507,0,0,0,0,0,93,348,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
531,0,0,0,0,0,93,349,0,64,1,1,5,105,110,77,83,69,1,0,3329,3,0,0,
555,0,0,0,0,0,93,350,0,64,1,1,5,105,110,77,83,69,1,0,3329,4,0,0,
579,0,0,0,0,0,93,351,0,64,1,1,5,105,110,77,83,69,1,0,3329,4,0,0,
603,0,0,9,0,0,17,352,0,0,1,1,0,1,0,1,97,0,0,
622,0,0,9,0,0,23,353,0,64,0,0,5,105,110,77,83,69,1,0,
642,0,0,0,0,0,91,362,0,64,0,0,5,105,110,77,83,69,1,0,
662,0,0,0,0,0,90,426,0,64,0,0,6,105,110,67,73,78,78,1,0,
683,0,0,0,0,0,93,427,0,64,0,0,6,105,110,67,73,78,78,1,0,
704,0,0,0,0,0,93,428,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
729,0,0,0,0,0,93,429,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
754,0,0,0,0,0,93,430,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
779,0,0,0,0,0,93,431,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,3,0,0,
804,0,0,0,0,0,93,432,0,64,1,1,6,105,110,67,73,78,78,1,0,3329,4,0,0,
829,0,0,9,0,0,17,433,0,0,1,1,0,1,0,1,97,0,0,
848,0,0,9,0,0,23,434,0,64,0,0,6,105,110,67,73,78,78,1,0,
869,0,0,0,0,0,91,442,0,64,0,0,6,105,110,67,73,78,78,1,0,
890,0,0,10,0,0,17,474,0,0,1,1,0,1,0,1,97,0,0,
909,0,0,10,0,0,45,476,0,0,0,0,0,1,0,
924,0,0,10,0,0,13,478,0,0,2,0,0,1,0,2,3,0,0,2,4,0,0,
947,0,0,10,0,0,15,489,0,0,0,0,0,1,0,
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
void LogWrite(tDebugInfo* DebugParms, int LogType, const char* msg, int argcount, ...) {
	// pLogLevel=	0 (No screen, No file) | 1 (Screen, No file) | 2 (Screen AND File)
	int n;
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	va_list			arguments;
	char submsg[MAX_PATH];
	unsigned int	im = 0;
	int				prev_im = 0;
	char timestamp[60];

	if (DebugParms->DebugLevel==0&&LogType==LOG_INFO) return;

	if (DebugParms->ThreadSafeLogging>0) WaitForSingleObject(DebugParms->Mtx, INFINITE);

	//-- Opens Log file only once
	if (DebugParms->fIsOpen!=1) {
		strcpy(DebugParms->FullfName, DebugParms->fPath); strcat(DebugParms->FullfName, "/"); strcat(DebugParms->FullfName, DebugParms->fName);
		DebugParms->fHandle = fopen(DebugParms->FullfName, "a");
		DebugParms->fIsOpen = 1;
		getCurrentTime(timestamp);
		fprintf(DebugParms->fHandle, "\n---------- Process %d Started New Log at %s ----------\n", GetCurrentProcessId(), timestamp);
	}

	va_start(arguments, argcount);
	n = 0;

	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			n++;
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				if (DebugParms->DebugLevel==1||DebugParms->DebugLevel==3||LogType==LOG_ERROR) printf(submsg, arg_s);
				if (DebugParms->DebugLevel==2||DebugParms->DebugLevel==3||LogType==LOG_ERROR)	fprintf(DebugParms->fHandle, submsg, arg_s);
			} else if (msg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				if (DebugParms->DebugLevel==1||DebugParms->DebugLevel==3||LogType==LOG_ERROR) printf(submsg, arg_d);
				if (DebugParms->DebugLevel==2||DebugParms->DebugLevel==3||LogType==LOG_ERROR)	fprintf(DebugParms->fHandle, submsg, arg_d);
			} else if (msg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				if (DebugParms->DebugLevel==1||DebugParms->DebugLevel==3||LogType==LOG_ERROR) printf(submsg, arg_d);
				if (DebugParms->DebugLevel==2||DebugParms->DebugLevel==3||LogType==LOG_ERROR)	fprintf(DebugParms->fHandle, submsg, arg_d);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				if (DebugParms->DebugLevel==1||DebugParms->DebugLevel==3||LogType==LOG_ERROR) printf(submsg, arg_f);
				if (DebugParms->DebugLevel==2||DebugParms->DebugLevel==3||LogType==LOG_ERROR)	fprintf(DebugParms->fHandle, submsg, arg_f);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	if (DebugParms->DebugLevel==1||DebugParms->DebugLevel==3||LogType==LOG_ERROR) printf(submsg);
	if (DebugParms->DebugLevel==2||DebugParms->DebugLevel==3||LogType==LOG_ERROR) fprintf(DebugParms->fHandle, submsg);
	if (LogType==LOG_ERROR && DebugParms->PauseOnError>0) { printf("Press any key..."); getchar(); }

	va_end(arguments);

	if (DebugParms->ThreadSafeLogging>0) ReleaseMutex(DebugParms->Mtx);
}

//=== DB common functions
EXPORT int  OraConnect(tDebugInfo* DebugParms, tDBConnection* DBConnInfo) {

	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	char* username = DBConnInfo->DBUser;
	char* password = DBConnInfo->DBPassword;
	char* dbstring = DBConnInfo->DBConnString;
	sql_context vCtx = NULL;
	/* EXEC SQL END DECLARE SECTION; */ 


	//LogWrite_C(DebugParms, "%s About to execute connect with %s , %s , %s\n", 4, timestamp_C, pUserName, pPassword, pDBString);

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
		//DebugParms->DBCtx = vCtx;
		LogWrite(DebugParms, LOG_INFO, "OraConnect() - Connected to ORACLE as user: %s ; DBConnInfo->DBCtx=%p\n", 2, username, DBConnInfo->DBCtx);
	} else {
		LogWrite(DebugParms, LOG_INFO, "PATH=%s\n", 1, vPath);
		LogWrite(DebugParms, LOG_INFO, "ORACLE_HOME=%s\n", 1, vOH);
		LogWrite(DebugParms, LOG_INFO, "DBUser=%s\n", 1, DBConnInfo->DBUser);
		//Oerr(DebugParms, __func__, sqlca.sqlcode);
		LogWrite(DebugParms, LOG_ERROR, "%s failed. user = %s\n Error %s", 3, __func__, DBConnInfo->DBUser, sqlca.sqlerrm.sqlerrmc);
	}
	return(sqlca.sqlcode);
}
EXPORT void OraDisconnect(sql_context pCtx, int Commit) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = pCtx;
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
EXPORT void OraCommit(void* pCtx) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	sql_context vCtx = pCtx;
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
EXPORT int GetFlatBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, float* oBarData, float* oBaseBar) {
	//-- 'Flat' bar means returning OHLC(V) values for each record as an array 
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int retval = 0;
	int i;
	char stmt[1000];
	char vNDT[12+1];
	float vOpen, vHigh, vLow, vClose;
	/* EXEC SQL END   DECLARE SECTION; */ 

	strcpy(stmt, pSQL);

	/* EXEC SQL CONTEXT USE : pCtx; */ 

	/* EXEC SQL PREPARE s4f FROM : stmt; */ 

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
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL DECLARE c4f CURSOR FOR s4f; */ 

	/* EXEC SQL OPEN c4f; */ 

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
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	for (i = pRecCount-1; i>=0; i--) {
		/* EXEC SQL FETCH c4f into : vNDT, : vOpen, : vHigh, : vLow, : vClose; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 5;
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
  sqlstm.sqphsv = sqlstm.sqhstv;
  sqlstm.sqphsl = sqlstm.sqhstl;
  sqlstm.sqphss = sqlstm.sqhsts;
  sqlstm.sqpind = sqlstm.sqindv;
  sqlstm.sqpins = sqlstm.sqinds;
  sqlstm.sqparm = sqlstm.sqharm;
  sqlstm.sqparc = sqlstm.sqharc;
  sqlstm.sqpadto = sqlstm.sqadto;
  sqlstm.sqptdso = sqlstm.sqtdso;
  sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


		if (sqlca.sqlcode==0) {
			oBarData[4*i+0] = vOpen;
			oBarData[4*i+1] = vHigh;
			oBarData[4*i+2] = vLow;
			oBarData[4*i+3] = vClose;
		} else if (sqlca.sqlcode==1403) {
			break;
		} else {
			LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
			retval = sqlca.sqlcode;
			break;
		}
	}
	//-- one more fetch to get baseBar
	/* EXEC SQL FETCH c4f into : vNDT, : vOpen, : vHigh, : vLow, : vClose; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 5;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )169;
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
 sqlstm.sqphsv = sqlstm.sqhstv;
 sqlstm.sqphsl = sqlstm.sqhstl;
 sqlstm.sqphss = sqlstm.sqhsts;
 sqlstm.sqpind = sqlstm.sqindv;
 sqlstm.sqpins = sqlstm.sqinds;
 sqlstm.sqparm = sqlstm.sqharm;
 sqlstm.sqparc = sqlstm.sqharc;
 sqlstm.sqpadto = sqlstm.sqadto;
 sqlstm.sqptdso = sqlstm.sqtdso;
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	if (sqlca.sqlcode==0) {
		oBaseBar[0] = vOpen;
		oBaseBar[1] = vHigh;
		oBaseBar[2] = vLow;
		oBaseBar[3] = vClose;
	} else if (sqlca.sqlcode!=1403) {
		LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		retval = sqlca.sqlcode;
	}

	/* EXEC SQL CLOSE c4f; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 5;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )204;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	return retval;
}
EXPORT int GetBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, int pSkipFirstN, tBar* oBar) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i = 0;  int j;
	int retval = 0;
	char vNDT[12+1];
	//sql_context vCtx = pCtx;
	struct sBar {
		char NewDateTime[12+1];
		double Open;
		double High;
		double Low;
		double Close;
		double OpenD;
		double HighD;
		double LowD;
		double CloseD;
		double Volume;
		double VolumeD;
	} *vBar;
	char stmt[1000];
	/* EXEC SQL END DECLARE SECTION; */ 

	vBar = (struct sBar*)malloc((pRecCount+pSkipFirstN)*sizeof(struct sBar));
	strcpy(stmt, pSQL);

	/* EXEC SQL CONTEXT USE : pCtx; */ 

	/* EXEC SQL PREPARE s4 FROM : stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 5;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )219;
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
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	/* EXEC SQL DECLARE c4 CURSOR FOR s4; */ 

	/* EXEC SQL OPEN c4; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 5;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )238;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqcmod = (unsigned int )0;
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	for (i = 0; i<(pRecCount+pSkipFirstN); i++) {
		/* EXEC SQL FETCH c4 INTO : vNDT, : vBar[i].Open, : vBar[i].High, : vBar[i].Low, : vBar[i].Close, : vBar[i].Volume; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 6;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )253;
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
  sqlstm.sqhstv[1] = (         void  *)&(vBar[i].Open);
  sqlstm.sqhstl[1] = (unsigned int  )sizeof(double);
  sqlstm.sqhsts[1] = (         int  )0;
  sqlstm.sqindv[1] = (         void  *)0;
  sqlstm.sqinds[1] = (         int  )0;
  sqlstm.sqharm[1] = (unsigned int  )0;
  sqlstm.sqadto[1] = (unsigned short )0;
  sqlstm.sqtdso[1] = (unsigned short )0;
  sqlstm.sqhstv[2] = (         void  *)&(vBar[i].High);
  sqlstm.sqhstl[2] = (unsigned int  )sizeof(double);
  sqlstm.sqhsts[2] = (         int  )0;
  sqlstm.sqindv[2] = (         void  *)0;
  sqlstm.sqinds[2] = (         int  )0;
  sqlstm.sqharm[2] = (unsigned int  )0;
  sqlstm.sqadto[2] = (unsigned short )0;
  sqlstm.sqtdso[2] = (unsigned short )0;
  sqlstm.sqhstv[3] = (         void  *)&(vBar[i].Low);
  sqlstm.sqhstl[3] = (unsigned int  )sizeof(double);
  sqlstm.sqhsts[3] = (         int  )0;
  sqlstm.sqindv[3] = (         void  *)0;
  sqlstm.sqinds[3] = (         int  )0;
  sqlstm.sqharm[3] = (unsigned int  )0;
  sqlstm.sqadto[3] = (unsigned short )0;
  sqlstm.sqtdso[3] = (unsigned short )0;
  sqlstm.sqhstv[4] = (         void  *)&(vBar[i].Close);
  sqlstm.sqhstl[4] = (unsigned int  )sizeof(double);
  sqlstm.sqhsts[4] = (         int  )0;
  sqlstm.sqindv[4] = (         void  *)0;
  sqlstm.sqinds[4] = (         int  )0;
  sqlstm.sqharm[4] = (unsigned int  )0;
  sqlstm.sqadto[4] = (unsigned short )0;
  sqlstm.sqtdso[4] = (unsigned short )0;
  sqlstm.sqhstv[5] = (         void  *)&(vBar[i].Volume);
  sqlstm.sqhstl[5] = (unsigned int  )sizeof(double);
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
  sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}


		if (sqlca.sqlcode==0) {
			strcpy(vBar[i].NewDateTime, vNDT);
		} else if (sqlca.sqlcode==1403) {
			break;
		} else {
			LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
			retval = sqlca.sqlcode;
			break;
		}
	}
	/* EXEC SQL CLOSE c4; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )292;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&pCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	if (i > 0) {
		for (j = 0; j < pRecCount; j++) {
			memcpy(&oBar[j], &vBar[pSkipFirstN+j], sizeof(tBar));
		}
	}
	free(vBar);
	return retval;
}
EXPORT int GetCharPFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, char* oRet) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	char stmt[1000];
	char vRet[1000];
	sql_context vCtx = pCtx;
	/* EXEC SQL END   DECLARE SECTION; */ 

	strcpy(stmt, pSQL);

	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL PREPARE s8 FROM : stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )307;
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


	/* EXEC SQL DECLARE c8 CURSOR FOR s8; */ 

	/* EXEC SQL OPEN c8; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )326;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqcmod = (unsigned int )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	if (sqlca.sqlcode!=0) {
		LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	/* EXEC SQL FETCH c8 INTO : vRet; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )341;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqfoff = (           int )0;
 sqlstm.sqfmod = (unsigned int )2;
 sqlstm.sqhstv[0] = (         void  *)vRet;
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
		LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	Trim(vRet); strcpy(oRet, vRet);
	/* EXEC SQL CLOSE c8; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )360;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	return sqlca.sqlcode;
}
EXPORT int GetStringArrayFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int ArrLen, char** oRet) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	char stmt[1000];
	char vRet[1000];
	sql_context vCtx = pCtx;
	/* EXEC SQL END   DECLARE SECTION; */ 

	strcpy(stmt, pSQL);

	/* EXEC SQL CONTEXT USE : vCtx; */ 

	/* EXEC SQL PREPARE s82 FROM : stmt; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )375;
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


	/* EXEC SQL DECLARE c82 CURSOR FOR s82; */ 

	/* EXEC SQL OPEN c82; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.stmt = "";
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )394;
 sqlstm.selerr = (unsigned short)1;
 sqlstm.sqlpfmem = (unsigned int  )0;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlstm.sqcmod = (unsigned int )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	if (sqlca.sqlcode!=0) {
		LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
		return sqlca.sqlcode;
	}
	for (int i=0; i<ArrLen; i++) {
		/* EXEC SQL FETCH c82 INTO : vRet; */ 

{
  struct sqlexd sqlstm;
  sqlstm.sqlvsn = 13;
  sqlstm.arrsiz = 6;
  sqlstm.sqladtp = &sqladt;
  sqlstm.sqltdsp = &sqltds;
  sqlstm.iters = (unsigned int  )1;
  sqlstm.offset = (unsigned int  )409;
  sqlstm.selerr = (unsigned short)1;
  sqlstm.sqlpfmem = (unsigned int  )0;
  sqlstm.cud = sqlcud0;
  sqlstm.sqlest = (unsigned char  *)&sqlca;
  sqlstm.sqlety = (unsigned short)4352;
  sqlstm.occurs = (unsigned int  )0;
  sqlstm.sqfoff = (           int )0;
  sqlstm.sqfmod = (unsigned int )2;
  sqlstm.sqhstv[0] = (         void  *)vRet;
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
			LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
			return sqlca.sqlcode;
		}
		Trim(vRet); strcpy(oRet[i], vRet);
	}
	/* EXEC SQL CLOSE c82; */ 

{
 struct sqlexd sqlstm;
 sqlstm.sqlvsn = 13;
 sqlstm.arrsiz = 6;
 sqlstm.sqladtp = &sqladt;
 sqlstm.sqltdsp = &sqltds;
 sqlstm.iters = (unsigned int  )1;
 sqlstm.offset = (unsigned int  )428;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	return sqlca.sqlcode;
}

//=== Logging functions
EXPORT int Ora_LogSaveMSE(tDebugInfo* DebugParms, int pid, int tid, int mseCnt, float* mseT, float* mseV) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int vInsertCount;
	sql_context vCtx = DebugParms->DebugDB->DBCtx;
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
		if (OraConnect(DebugParms, DebugParms->DebugDB)!=0) {
			LogWrite(DebugParms, LOG_ERROR, "%s() could not connect to Log Database...\n", 1, __func__);
			return -1;
		}
		vCtx = DebugParms->DebugDB->DBCtx;
	}
	LogWrite(DebugParms, LOG_INFO, "%s() CheckPoint 2 - LogDB->DBCtx=%p , vCtx=%p\n", 3, __func__, DebugParms->DebugDB->DBCtx, vCtx);

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
	sprintf(&stmt[0], "insert into MyLog_MSE (ProcessId, ThreadId, Epoch, MSE_T, MSE_V) values(:P01, :P02, :P03, :P04, :P05)");
	LogWrite(DebugParms, LOG_INFO, "%s() CheckPoint 4 - stmt='%s' ; pid[0]=%d ; tid[0]=%d\n", 4, __func__, stmt, pid, tid);

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
 sqlstm.offset = (unsigned int  )443;
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
 sqlstm.offset = (unsigned int  )463;
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
 sqlstm.offset = (unsigned int  )483;
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
 sqlstm.offset = (unsigned int  )507;
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
 sqlstm.offset = (unsigned int  )531;
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
 sqlstm.offset = (unsigned int  )555;
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
 sqlstm.offset = (unsigned int  )579;
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
 sqlstm.offset = (unsigned int  )603;
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
 sqlstm.offset = (unsigned int  )622;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	//EXEC SQL ALTER SESSION SET SQL_TRACE = FALSE;

	//pInsertCount[vNetLevel] = sqlca.sqlerrd[2];
	LogWrite(DebugParms, LOG_INFO, "BulkMSEInsert() inserted %d rows.\n", 1, sqlca.sqlerrd[2]);
	if (sqlca.sqlcode!=0) {
		if (sqlca.sqlcode!=1) LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
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
 sqlstm.offset = (unsigned int  )642;
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
EXPORT int Ora_LogSaveW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int vInsertCount;
	sql_context vCtx = DebugParms->DebugDB->DBCtx;
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
		if (OraConnect(DebugParms, DebugParms->DebugDB)!=0) {
			LogWrite(DebugParms, LOG_ERROR, "%s() could not connect to Log Database...\n", 1, __func__);
			return -1;
		}
		vCtx = DebugParms->DebugDB->DBCtx;
	}
	LogWrite(DebugParms, LOG_INFO, "%s() CheckPoint 2 - LogDB->DBCtx=%p , vCtx=%p\n", 3, __func__, DebugParms->DebugDB->DBCtx, vCtx);

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
	LogWrite(DebugParms, LOG_INFO, "%s() CheckPoint 4 - stmt='%s' ; pid[0]=%d ; tid[0]=%d\n", 4, __func__, stmt, pid, tid);

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
 sqlstm.offset = (unsigned int  )662;
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
 sqlstm.offset = (unsigned int  )683;
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
 sqlstm.offset = (unsigned int  )704;
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
 sqlstm.offset = (unsigned int  )729;
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
 sqlstm.offset = (unsigned int  )754;
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
 sqlstm.offset = (unsigned int  )779;
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
 sqlstm.offset = (unsigned int  )804;
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
 sqlstm.offset = (unsigned int  )829;
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
 sqlstm.offset = (unsigned int  )848;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}


	//EXEC SQL ALTER SESSION SET SQL_TRACE = FALSE;

	LogWrite(DebugParms, LOG_INFO, "%s() inserted %d rows.\n", 2, __func__, sqlca.sqlerrd[2]);
	if (sqlca.sqlcode!=0) {
		if (sqlca.sqlcode!=1) LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
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
 sqlstm.offset = (unsigned int  )869;
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

EXPORT int Ora_LogLoadW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* oW) {
	/* EXEC SQL BEGIN DECLARE SECTION; */ 

	int i;
	int retval = 0;
	sql_context vCtx = DebugParms->DebugDB->DBCtx;
	char stmt[1000];
	//-- Network Weights variables
	int vWid; double vW;
	/* EXEC SQL END DECLARE SECTION; */ 


	//-- Connects to DB only once
	if (vCtx==NULL) {
		if (OraConnect(DebugParms, DebugParms->DebugDB)!=0) printf("%s() could not connect to Log Database...\n", __func__);
		vCtx = DebugParms->DebugDB->DBCtx;
	}

	//-- Load Network Weights from CoreImage_NN
	sprintf(&stmt[0], "select WId, W from CoreImage_NN where ProcessId=%d and ThreadId=%d and Epoch=%d order by 1,2,3", pid, tid, epoch);
	LogWrite(DebugParms, LOG_INFO, "%s() CheckPoint 3 - stmt=%s\n", 2, __func__, stmt);
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
 sqlstm.offset = (unsigned int  )890;
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
 sqlstm.offset = (unsigned int  )909;
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
  sqlstm.offset = (unsigned int  )924;
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
			oW[i] = vW;
		} else if (sqlca.sqlcode==1403) {
			break;
		} else {
			LogWrite(DebugParms, LOG_ERROR, "%s failed. stmt = %s\n Error %s", 3, __func__, stmt, sqlca.sqlerrm.sqlerrmc);
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
 sqlstm.offset = (unsigned int  )947;
 sqlstm.cud = sqlcud0;
 sqlstm.sqlest = (unsigned char  *)&sqlca;
 sqlstm.sqlety = (unsigned short)4352;
 sqlstm.occurs = (unsigned int  )0;
 sqlcxt(&vCtx, &sqlctx, &sqlstm, &sqlfpn);
}



	return retval;
}