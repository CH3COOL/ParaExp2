#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"Windows.h"
#include <xmmintrin.h>//SSE
typedef float dType;
void NormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	for(int i=0;i<n*m;i++)res[i]=0;
//	memset(res,0,n*m*4);
	for(int i=0;i<n;i++)
	for(int j=0;j<m;j++)
	for(int k=0;k<s;k+=4)
	{
		res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
		res[n*i+j]+=(a[n*i+k+1]*b[s*(k+1)+j]);
		res[n*i+j]+=(a[n*i+k+2]*b[s*(k+2)+j]);
		res[n*i+j]+=(a[n*i+k+3]*b[s*(k+3)+j]);
	}
}
void cacheNormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	for(int i=0;i<n*m;i++)res[i]=0;
//	memset(res,0,n*m*4);
	for(int i=0;i<n;i++)
	for(int k=0;k<s;k++)
	for(int j=0;j<m;j+=4)
	{
		res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
		res[n*i+j+1]+=(a[n*i+k]*b[s*k+j+1]);
		res[n*i+j+2]+=(a[n*i+k]*b[s*k+j+2]);
		res[n*i+j+3]+=(a[n*i+k]*b[s*k+j+3]);
	}
}
void MatAdd(int n,int m,dType*a,dType*b,dType*res)
{
	for(int i=0;i<n*m;i+=4)
	{
		res[i]=a[i]+b[i];
		res[i+1]=a[i+1]+b[i+1];
		res[i+2]=a[i+2]+b[i+2];
		res[i+3]=a[i+3]+b[i+3];
	}
}
void MatSub(int n,int m,dType*a,dType*b,dType*res)
{
	for(int i=0;i<n*m;i+=4)
	{
		res[i]=a[i]-b[i];
		res[i+1]=a[i+1]-b[i+1];
		res[i+2]=a[i+2]-b[i+2];
		res[i+3]=a[i+3]-b[i+3];
	}
}
void Strassen(int N,dType*a,dType*b,dType*res)
{
	if(N<8)
	{
		for(int i=0;i<N*N;i++)res[i]=0;
		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j+=4)
		{
			res[N*i+j]+=(a[N*i+k]*b[N*k+j]);
			res[N*i+j+1]+=(a[N*i+k]*b[N*k+j+1]);
			res[N*i+j+2]+=(a[N*i+k]*b[N*k+j+2]);
			res[N*i+j+3]+=(a[N*i+k]*b[N*k+j+3]);
		}
		return;
	}
	dType*a11=new dType[N*N/4];
	dType*a12=new dType[N*N/4];
	dType*a21=new dType[N*N/4];
	dType*a22=new dType[N*N/4];
	dType*b11=new dType[N*N/4];
	dType*b12=new dType[N*N/4];
	dType*b21=new dType[N*N/4];
	dType*b22=new dType[N*N/4];
	
	dType*a12Sa22=new dType[N*N/4];//1
	dType*b21Ab22=new dType[N*N/4];//2
	dType*a11Aa22=new dType[N*N/4];//3
	dType*b11Ab22=new dType[N*N/4];//4
	dType*a21Sa11=new dType[N*N/4];//5
	dType*b11Ab12=new dType[N*N/4];//6
	dType*a11Aa12=new dType[N*N/4];//7
	dType*b12Sb22=new dType[N*N/4];//8
	dType*b21Sb11=new dType[N*N/4];//9
	dType*a21Aa22=new dType[N*N/4];//10
	
	dType*M1=new dType[N*N/4];
	dType*M2=new dType[N*N/4];
	dType*M3=new dType[N*N/4];
	dType*M4=new dType[N*N/4];
	dType*M5=new dType[N*N/4];
	dType*M6=new dType[N*N/4];
	dType*M7=new dType[N*N/4];
	for(int i=0;i<N/2;i++)
	{
		for(int j=0;j<N/2;j++)
		{
			a11[(N/2)*i+j]=a[N*i+j];
			b11[(N/2)*i+j]=b[N*i+j];
		}
		for(int j=N/2;j<N;j++)
		{
			a12[(N/2)*i+j-N/2]=a[N*i+j];
			b12[(N/2)*i+j-N/2]=b[N*i+j];
		}
	}
	for(int i=N/2;i<N;i++)
	{
		for(int j=0;j<N/2;j++)
		{
			a21[(N/2)*(i-N/2)+j]=a[N*i+j];
			b21[(N/2)*(i-N/2)+j]=b[N*i+j];
		}
		for(int j=N/2;j<N;j++)
		{
			a22[(N/2)*(i-N/2)+j-N/2]=a[N*i+j];
			b22[(N/2)*(i-N/2)+j-N/2]=b[N*i+j];
		}
	}
	MatSub(N/2,N/2,a12,a22,a12Sa22);
	MatAdd(N/2,N/2,b21,b22,b21Ab22);
	MatAdd(N/2,N/2,a11,a22,a11Aa22);
	MatAdd(N/2,N/2,b11,b22,b11Ab22);
	MatSub(N/2,N/2,a21,a11,a21Sa11);
	MatAdd(N/2,N/2,b11,b12,b11Ab12);
	MatAdd(N/2,N/2,a11,a12,a11Aa12);
	MatSub(N/2,N/2,b12,b22,b12Sb22);
	MatSub(N/2,N/2,b21,b11,b21Sb11);
	MatAdd(N/2,N/2,a21,a22,a21Aa22);
	
	Strassen(N/2,a12Sa22,b21Ab22,M1);
	Strassen(N/2,a11Aa22,b11Ab22,M2);
	Strassen(N/2,a21Sa11,b11Ab12,M3);
	Strassen(N/2,a11Aa12,b22,M4);
	Strassen(N/2,a11,b12Sb22,M5);
	Strassen(N/2,a22,b21Sb11,M6);
	Strassen(N/2,a21Aa22,b11,M7);
	
	MatAdd(N/2,N/2,M1,M2,M1);
	MatAdd(N/2,N/2,M1,M6,M1);
	MatSub(N/2,N/2,M1,M4,M1);//C11=M1
	
	MatAdd(N/2,N/2,M4,M5,M4);//C12=M4
	
	MatAdd(N/2,N/2,M6,M7,M6);//C21=M6
	
	MatAdd(N/2,N/2,M2,M3,M2);
	MatAdd(N/2,N/2,M2,M5,M2);
	MatSub(N/2,N/2,M2,M7,M2);//C22=M2
	
	for(int i=0;i<N/2;i++)
	{
		for(int j=0;j<N/2;j++)
			res[N*i+j]=M1[(N/2)*i+j];
		for(int j=N/2;j<N;j++)
			res[N*i+j]=M4[(N/2)*i+j-N/2];
	}
	for(int i=N/2;i<N;i++)
	{
		for(int j=0;j<N/2;j++)
			res[N*i+j]=M6[(N/2)*(i-N/2)+j];
		for(int j=N/2;j<N;j++)
			res[N*i+j]=M2[(N/2)*(i-N/2)+j-N/2];
	}
	delete[]a11;
	delete[]a12;
	delete[]a21;
	delete[]a22;
	delete[]b11;
	delete[]b12;
	delete[]b21;
	delete[]b22;
	delete[]a12Sa22;
	delete[]b21Ab22;
	delete[]a11Aa22;
	delete[]b11Ab22;
	delete[]a21Sa11;
	delete[]b11Ab12;
	delete[]a11Aa12;
	delete[]b12Sb22;
	delete[]b21Sb11;
	delete[]a21Aa22;
	delete[]M1;
	delete[]M2;
	delete[]M3;
	delete[]M4;
	delete[]M5;
	delete[]M6;
	delete[]M7;
}
void MatAdd(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=4)
	{
		res[i*resRowLen+j]=a[i*aRowLen+j]+b[i*bRowLen+j];
		res[i*resRowLen+j+1]=a[i*aRowLen+j+1]+b[i*bRowLen+j+1];
		res[i*resRowLen+j+2]=a[i*aRowLen+j+2]+b[i*bRowLen+j+2];
		res[i*resRowLen+j+3]=a[i*aRowLen+j+3]+b[i*bRowLen+j+3];
	}
}
void MatSub(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=4)
	{
		res[i*resRowLen+j]=a[i*aRowLen+j]-b[i*bRowLen+j];
		res[i*resRowLen+j+1]=a[i*aRowLen+j+1]-b[i*bRowLen+j+1];
		res[i*resRowLen+j+2]=a[i*aRowLen+j+2]-b[i*bRowLen+j+2];
		res[i*resRowLen+j+3]=a[i*aRowLen+j+3]-b[i*bRowLen+j+3];
	}
}
void Strassen(int N,
				int aRowLen,const dType*a,
				int bRowLen,const dType*b,
				int resRowLen,dType*res)
{
	if(N<8)
	{
		for(int i=0;i<N;i++)
		for(int j=0;j<N;j+=4)
		{
			res[resRowLen*i+j]=0;
			res[resRowLen*i+j+1]=0;
			res[resRowLen*i+j+2]=0;
			res[resRowLen*i+j+3]=0;
		}

		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j+=4)
		{
			res[resRowLen*i+j]+=(a[aRowLen*i+k]*b[bRowLen*k+j]);
			res[resRowLen*i+j+1]+=(a[aRowLen*i+k]*b[bRowLen*k+j+1]);
			res[resRowLen*i+j+2]+=(a[aRowLen*i+k]*b[bRowLen*k+j+2]);
			res[resRowLen*i+j+3]+=(a[aRowLen*i+k]*b[bRowLen*k+j+3]);
		}
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
		(dType*)malloc(sz),
		(dType*)malloc(sz),
		(dType*)malloc(sz),
		(dType*)malloc(sz),
		(dType*)malloc(sz),
		(dType*)malloc(sz),
		(dType*)malloc(sz)
	};
	dType*T=(dType*)malloc(sz);
  	dType*U=(dType*)malloc(sz);
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
	
	free(U);  // deallocate temp matrices
	free(T);
	free(P[0]);
	free(P[1]);
	free(P[2]);
	free(P[3]);
	free(P[4]);
	free(P[5]);
	free(P[6]);
}
int main()
{
	srand((unsigned)time(NULL));
    int N=2048;
    dType*a=new dType[N*N];
    for(int i=0;i<N*N;i++)a[i]=i;
    dType*b=new dType[N*N];
    for(int i=0;i<N*N;i++)b[i]=i;
    dType*c1=new dType[N*N];
    dType*c2=new dType[N*N];
    dType*c3=new dType[N*N];
    long long head,tail,freq;
    float tN,tC,tS;
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
    //Strassen(N,a,b,c3);
    Strassen(N,N,a,N,b,N,c3);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    tS=(tail-head)*1000.0/freq;
    
    printf("N=%d\n",N);
    printf("Normal_SSE elapsed %.3f ms\n",tN);
    printf("Cache_SSE elasped %.3f ms\n",tC);
    printf("Strassen_SSE Opti elasped %.3f ms\n",tS);
    //freopen("c1.txt","w",stdout);
//    freopen("Log44.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//    if(!(c1[i]==c2[i]&&c2[i]==c3[i]))
//	{
//		printf("Result Inconsistent at %d,%d\n",i/N,i%N);
//		printf("c1=%f c2=%f c3=%f\n",c1[i],c2[i],c3[i]);
//		printf("rate=%f\n\n",(float)c1[i]/c3[i]);
//	}
//	freopen("c2.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//		printf("%d\n",c2[i]);
//	freopen("c3.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//		printf("%d\n",c3[i]);
    delete[]a;
    delete[]b;
    delete[]c1;
    delete[]c2;
    delete[]c3;
    return 0;
}
