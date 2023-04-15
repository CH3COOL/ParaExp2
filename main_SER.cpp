#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"Windows.h"
#define MALLOC_SPACE(NBytes) (dType*)malloc(NBytes)
#define FREE_SPACE(Ptr) free(Ptr)
#define SLI 64
typedef float dType;
void NormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	int nm=n*m;
	for(int i=0;i<nm;i++)res[i]=0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			//int ni=n*i;
			//int nij=n*i+j;
			for(int k=0;k<s;k++)
			{
				res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
			}
		}
	}
}
void cacheNormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	for(int i=0;i<n*m;i++)res[i]=0;
	for(int i=0;i<n;i++)
		for(int k=0;k<s;k++)
			for(int j=0;j<m;j++)
				res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
}
void slicePolicy(int N,dType*a,dType*b,dType*res,int Block_SZ)
{
	int blockCnt=N/Block_SZ;
	int NN=N*N; 
	for(int i=0;i<NN;i++)res[i]=0;
	for(int bi=0;bi<blockCnt;bi++)
		for(int bk=0;bk<blockCnt;bk++)
		{
			for(int bj=0;bj<blockCnt;bj++)//block_sz
				for(int i=0;i<Block_SZ;i++)
				{
					int idxI=N*(i+bi*Block_SZ);
					for(int k=0;k<Block_SZ;k++)
					{
						int idxK=(k+bk*Block_SZ);
						int idxNk=N*(k+bk*Block_SZ);
						for(int j=0;j<Block_SZ;j++)//inner mul
						{
							int idxJ=(j+bj*Block_SZ);
							res[idxI+idxJ]+=
								a[idxI+idxK]*
								b[idxNk+idxJ];
						}
					}
				}
		}
}
void transposePolicy(int n,int s,int m,dType*a,dType*b,dType*res)
{
	dType*bT=MALLOC_SPACE(s*m*sizeof(dType));
	for(int i=0;i<n*m;i++)res[i]=0;
	for(int i=0;i<s;i++)
		for(int j=0;j<m;j++)
			bT[j*m+i]=b[i*s+j];
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			for(int k=0;k<s;k++)
				res[n*i+j]+=(a[n*i+k]*bT[m*j+k]);
	FREE_SPACE(bT);
}
void MatAdd(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j++)
	res[i*resRowLen+j]=a[i*aRowLen+j]+b[i*bRowLen+j];
}
void MatSub(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j++)
	res[i*resRowLen+j]=a[i*aRowLen+j]-b[i*bRowLen+j];
}
void Strassen(int N,
				int aRowLen,const dType*a,
				int bRowLen,const dType*b,
				int resRowLen,dType*res)
{
	if(N<=8)
	{
		for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		res[resRowLen*i+j]=0;

		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j++)
		res[resRowLen*i+j]+=(a[aRowLen*i+k]*b[bRowLen*k+j]);
		return;
	}
	const int n=N/2;
	const dType*A=a;
	const dType*B=a+n;
	const dType*C=a+n*aRowLen;
	const dType*D=C+n;
	
	const dType*E=b;
	const dType*F=b+n;
	const dType*G=b+n*bRowLen;
	const dType*H=G+n;
	
	const int sz=n*n*sizeof(dType);
	dType*P[7]={
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz),
		MALLOC_SPACE(sz)
	};
	dType*T=MALLOC_SPACE(sz);
  	dType*U=MALLOC_SPACE(sz);
	MatSub(n,bRowLen,F,bRowLen,H,n,T);//T=F-H
	Strassen(n,aRowLen,A,n,T,n,P[0]);//P0=A*(F-H)
	
	MatAdd(n,aRowLen,A,aRowLen,B,n,T);//T=A+B
  	Strassen(n,n,T,bRowLen,H,n,P[1]);//P1=(A+B)*H
  	
  	MatAdd(n, aRowLen, C, aRowLen, D, n, T);
 	Strassen(n, n, T, bRowLen, E, n, P[2]);// P2 = (C + D)*E
 	
	MatSub(n, bRowLen, G, bRowLen, E, n, T);//T=G-E
	Strassen(n, aRowLen, D, n, T, n, P[3]);//P3=D*(G-E)
	
	MatAdd(n, aRowLen, A, aRowLen, D, n, T);//T=A+D
	MatAdd(n, bRowLen, E, bRowLen, H, n, U);//U=E+H
	Strassen(n, n, T, n, U, n, P[4]);// P4 = (A + D)*(E + H)
	
	MatSub(n, aRowLen, B, aRowLen, D, n, T);//T=B-D
	MatAdd(n, bRowLen, G, bRowLen, H, n, U);//U=G+H
	Strassen(n, n, T, n, U, n, P[5]);// P5 = (B - D)*(G + H)
	
	MatSub(n, aRowLen, A, aRowLen, C, n, T);//T=A-C
	MatAdd(n, bRowLen, E, bRowLen, F, n, U);//U=E+F
	Strassen(n, n, T, n, U, n, P[6]);// P6 = (A - C)*(E + F)
	
	// Z upper left = (P3 + P4) + (P5 - P1)
	MatAdd(n, n, P[4], n, P[3], n, T);
	MatSub(n, n, P[5], n, P[1], n, U);
	MatAdd(n, n, T, n, U, resRowLen, res);
	
	// Z lower left = P2 + P3
	MatAdd(n, n, P[2], n, P[3], resRowLen, res + n*resRowLen);
	
	// Z upper right = P0 + P1
	MatAdd(n, n, P[0], n, P[1], resRowLen, res + n);
	
	// Z lower right = (P0 + P4) - (P2 + P6)
	MatAdd(n, n, P[0], n, P[4], n, T);
	MatAdd(n, n, P[2], n, P[6], n, U);
	MatSub(n, n, T, n, U, resRowLen, res + n*(resRowLen + 1));
	
	FREE_SPACE(U);  // deallocate temp matrices
	FREE_SPACE(T);
	FREE_SPACE(P[0]);
	FREE_SPACE(P[1]);
	FREE_SPACE(P[2]);
	FREE_SPACE(P[3]);
	FREE_SPACE(P[4]);
	FREE_SPACE(P[5]);
	FREE_SPACE(P[6]);
}
int main()
{
	srand((unsigned)time(NULL));
    int N=16;
    int nsz=N*N*sizeof(dType);
    dType*a=MALLOC_SPACE(nsz);
    for(int i=0;i<N*N;i++)a[i]=i;
    dType*b=MALLOC_SPACE(nsz);
    for(int i=0;i<N*N;i++)b[i]=i;
    dType*c1=MALLOC_SPACE(nsz);
    dType*c2=MALLOC_SPACE(nsz);
    dType*c3=MALLOC_SPACE(nsz);
    dType*c4=MALLOC_SPACE(nsz);
    dType*c5=MALLOC_SPACE(nsz); 
    long long head,tail,freq;
    float tN,tC,tS,tT,tSli;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
    NormalMul(N,N,N,a,b,c1);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	tN=(tail-head)*1000.0/freq;
	
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
    cacheNormalMul(N,N,N,a,b,c2);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    tC=(tail-head)*1000.0/freq;
    
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Strassen(N,N,a,N,b,N,c3);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    tS=(tail-head)*1000.0/freq;
    
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    transposePolicy(N,N,N,a,b,c4);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    tT=(tail-head)*1000.0/freq;
    
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    slicePolicy(N,a,b,c5,N>SLI?SLI:N);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    tSli=(tail-head)*1000.0/freq;
    
    printf("N=%d\n",N);
    printf("Normal elapsed %.3f ms\n",tN);
    printf("Cache elasped %.3f ms\n",tC);
    printf("Strassen Opti elasped %.3f ms\n",tS);
    printf("Transpose elasped %.3f ms\n",tT);
    printf("Slice elasped %.3f ms\n",tSli);
    //freopen("c1.txt","w",stdout);
//    freopen("Log44.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//    if(!(c1[i]==c2[i]&&c2[i]==c3[i]))
//	{
//		printf("Result Inconsistent at %d,%d\n",i/N,i%N);
//		printf("c1=%f c2=%f c3=%f\n",c1[i],c2[i],c3[i]);
//		printf("rate=%f\n\n",(float)c1[i]/c3[i]);
//	}
	freopen("c1.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c1[i]);
	freopen("c2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c2[i]);
	freopen("c3.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c3[i]);
	freopen("c4.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c4[i]);
	freopen("c5.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c5[i]);
    FREE_SPACE(a);
    FREE_SPACE(b);
    FREE_SPACE(c1);
	FREE_SPACE(c2);
	FREE_SPACE(c3);
	FREE_SPACE(c4);
	FREE_SPACE(c5);
    return 0;
}
