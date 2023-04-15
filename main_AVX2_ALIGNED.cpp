#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"Windows.h"
#include<immintrin.h>//AVX AVX2 AVX512
#define ALIGN_N 32
#define SLI 8
typedef float dType;
void NormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
//	memset(res,0,n*m*4);
	__m256 atemp;
	__m256 btemp;
	__m256 multmp;
	__m256 restemp;
	__m128 s1;
	__m128 s2;
	restemp=_mm256_setzero_ps();
	for(int i=0;i<n*m;i+=8)
		_mm256_store_ps(res+i,restemp);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			restemp=_mm256_setzero_ps();
			for(int k=0;k<s;k+=8)
			{
				atemp=_mm256_load_ps(a+n*i+k);
				btemp=_mm256_set_ps(b[s*(k+7)+j],b[s*(k+6)+j],b[s*(k+5)+j],b[s*(k+4)+j],
				b[s*(k+3)+j],b[s*(k+2)+j],b[s*(k+1)+j],b[s*(k)+j]);
				multmp=_mm256_mul_ps(atemp,btemp);
				restemp=_mm256_add_ps(restemp,multmp);
			}
			s1=_mm256_extractf128_ps(restemp,0);
			s2=_mm256_extractf128_ps(restemp,1);
			s1=_mm_hadd_ps(s1,s2);
			s1=_mm_hadd_ps(s1,s1);
			s1=_mm_hadd_ps(s1,s1);
			_mm_store_ss(res+n*i+j,s1);
		}
	}
}
void cacheNormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	__m256 atemp;
	__m256 btemp;
	__m256 mulTemp;
	__m256 resTemp;
	resTemp=_mm256_setzero_ps(); 
	for(int i=0;i<n*m;i+=8)
		_mm256_store_ps(res+i,resTemp);
	for(int i=0;i<n;i++)
	{
		for(int k=0;k<s;k++)
		{
			for(int j=0;j<m;j+=8)
			{
				//atemp=_mm256_load1_ps(a+n*i+k);
				atemp=_mm256_broadcast_ss(a+n*i+k);
				//atemp=_mm256_set1_ps(a[n*i+k]);
				btemp=_mm256_load_ps(b+s*k+j);
				mulTemp=_mm256_mul_ps(atemp,btemp);
				resTemp=_mm256_load_ps(res+n*i+j);
				resTemp=_mm256_add_ps(resTemp,mulTemp);
				_mm256_store_ps(res+n*i+j,resTemp);
			}
		}
	}
}
void slicePolicy(int N,dType*a,dType*b,dType*res,int Block_SZ)
{
	int blockCnt=N/Block_SZ;
	__m256 restemp;
	__m256 atemp,btemp;
	restemp=_mm256_setzero_ps();
	for(int i=0;i<N*N;i+=8)
		_mm256_store_ps(res+i,restemp);
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
						for(int j=0;j<Block_SZ;j+=8)//inner mul
						{
							int idxJ=(j+bj*Block_SZ);
							//res[idxI+idxJ]+=a[idxI+idxK]*b[idxNk+idxJ];
							//atemp=_mm256_set1_ps(a[idxI+idxK]);
							atemp=_mm256_broadcast_ss(a+idxI+idxK);
							btemp=_mm256_load_ps(b+idxNk+idxJ);
							atemp=_mm256_mul_ps(atemp,btemp);
							restemp=_mm256_load_ps(res+idxI+idxJ);
							restemp=_mm256_add_ps(restemp,atemp);
							_mm256_store_ps(res+idxI+idxJ,restemp);
						}
					}
				}
		}
}
void transposePolicy(int n,int s,int m,dType*a,dType*b,dType*res)
{
	dType*bT=(dType*)_mm_malloc(s*m*sizeof(dType),ALIGN_N);
	__m256 restemp,atemp,btemp;
	__m128 s1,s2;
	restemp=_mm256_setzero_ps();
	for(int i=0;i<n*m;i+=8)
		_mm256_store_ps(res+i,restemp);
	for(int i=0;i<s;i++)
		for(int j=0;j<m;j++)
			bT[j*m+i]=b[i*s+j];
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
		{
			restemp=_mm256_setzero_ps();
			for(int k=0;k<s;k+=8)
			{
				atemp=_mm256_load_ps(a+n*i+k);
				btemp=_mm256_load_ps(bT+m*j+k);
				atemp=_mm256_mul_ps(atemp,btemp);
				restemp=_mm256_add_ps(restemp,atemp);
				//res[n*i+j]+=(a[n*i+k]*bT[m*j+k]);
			}
			s1=_mm256_extractf128_ps(restemp,0);
			s2=_mm256_extractf128_ps(restemp,1);
			s1=_mm_hadd_ps(s1,s2);
			s1=_mm_hadd_ps(s1,s1);
			s1=_mm_hadd_ps(s1,s1);
			_mm_store_ss(res+n*i+j,s1);
		}	
	_mm_free(bT);
}
void MatAdd(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	__m256 atemp;
	__m256 btemp;
	__m256 resTemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=8)
	{
		atemp=_mm256_load_ps(a+i*aRowLen+j);
		btemp=_mm256_load_ps(b+i*bRowLen+j);
		resTemp=_mm256_add_ps(atemp,btemp);
		_mm256_store_ps(res+i*resRowLen+j,resTemp);
	}
}
void MatSub(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
	__m256 atemp;
	__m256 btemp;
	__m256 resTemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=8)
	{
		atemp=_mm256_load_ps(a+i*aRowLen+j);
		btemp=_mm256_load_ps(b+i*bRowLen+j);
		resTemp=_mm256_sub_ps(atemp,btemp);
		_mm256_store_ps(res+i*resRowLen+j,resTemp);
	}
}
void Strassen(int N,
				int aRowLen,const dType*a,
				int bRowLen,const dType*b,
				int resRowLen,dType*res)
{
	if(N<=8)
	{
		__m256 atemp;
		__m256 btemp;
		__m256 multemp;
		__m256 restemp;
		restemp=_mm256_setzero_ps();
		for(int i=0;i<N;i++)
		for(int j=0;j<N;j+=8)
		{
			_mm256_store_ps(res+resRowLen*i+j,restemp);
		}

		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j+=8)
		{
			//atemp=_mm256_set1_ps(a[aRowLen*i+k]);
			atemp=_mm256_broadcast_ss(a+aRowLen*i+k);
			btemp=_mm256_load_ps(b+bRowLen*k+j);
			multemp=_mm256_mul_ps(atemp,btemp);
			restemp=_mm256_load_ps(res+resRowLen*i+j);
			restemp=_mm256_add_ps(restemp,multemp);
			_mm256_store_ps(res+resRowLen*i+j,restemp);
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
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N),
		(dType*)_mm_malloc(sz,ALIGN_N)
	};
	dType*T=(dType*)_mm_malloc(sz,ALIGN_N);
  	dType*U=(dType*)_mm_malloc(sz,ALIGN_N);
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
	
	_mm_free(U);  // deallocate temp matrices
	_mm_free(T);
	_mm_free(P[0]);
	_mm_free(P[1]);
	_mm_free(P[2]);
	_mm_free(P[3]);
	_mm_free(P[4]);
	_mm_free(P[5]);
	_mm_free(P[6]);
}
int main()
{
	srand((unsigned)time(NULL));
    int N=16;
    dType*a=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    for(int i=0;i<N*N;i++)a[i]=i;
    dType*b=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    for(int i=0;i<N*N;i++)b[i]=N*N-1-i;
    dType*c1=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c2=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c3=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c4=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c5=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
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
    printf("Normal_A2 elapsed %.3f ms\n",tN);
    printf("Cache_A2 elasped %.3f ms\n",tC);
    printf("Strassen_A2 elasped %.3f ms\n",tS);
    printf("Transpose_A2 elasped %.3f ms\n",tT);
    printf("Sliced_A2 64*64Block elasped %.3f ms\n",tSli);

    freopen("cc1_A2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c1[i]);
	freopen("cc2_A2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c2[i]);
	freopen("cc3_A2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c3[i]);
	freopen("cc4_A2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c4[i]);
	freopen("cc5_A2.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c5[i]);
	_mm_free(a);
	_mm_free(b);
	_mm_free(c1);
	_mm_free(c2);
	_mm_free(c3);
	_mm_free(c4);
	_mm_free(c5);
    return 0;
}
