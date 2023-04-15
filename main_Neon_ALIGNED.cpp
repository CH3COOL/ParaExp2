#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"Windows.h"
//#include<arm_neon.h>
#ifdef ARM_PLATFORM
#  include <arm_neon.h>
#else
#  include "NEON_2_SSE.h"
#endif
#define ALIGN_N 32
typedef float dType;
void NormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
//	memset(res,0,n*m*4);
//	__m256 atemp;
//	__m256 btemp;
//	__m256 multmp;
//	__m256 restemp;
//	__m128 s1;
//	__m128 s2;
	float32x4_t atemp,btemp,restemp;
	float32x2_t s1,s2;
	restemp=vdupq_n_f32(0.0);
	for(int i=0;i<n*m;i+=4)
		_mm256_store_ps(res+i,restemp);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			restemp=vdupq_n_f32(0.0);
			for(int k=0;k<s;k+=4)
			{
				//res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
				//res[n*i+j]+=(a[n*i+k+1]*b[s*(k+1)+j]);
				//res[n*i+j]+=(a[n*i+k+2]*b[s*(k+2)+j]);
				//res[n*i+j]+=(a[n*i+k+3]*b[s*(k+3)+j]);
				//atemp=_mm_loadu_ps(a+n*i);
//				atemp=_mm256_load_ps(a+n*i+k);
//				btemp=_mm256_set_ps(b[s*(k+7)+j],b[s*(k+6)+j],b[s*(k+5)+j],b[s*(k+4)+j],
//				b[s*(k+3)+j],b[s*(k+2)+j],b[s*(k+1)+j],b[s*(k)+j]);
//				multmp=_mm256_mul_ps(atemp,btemp);
//				restemp=_mm256_add_ps(restemp,multmp);

				float temp[]={b[s*(k)+j],b[s*(k+1)+j],b[s*(k+2)+j],b[s*(k+3)+j]};

				atemp=vld1q_f32(a+n*i+k);
				btemp=vld1q_f32(temp);
				atemp=vmulq_f32(atemp,btemp);
				restemp=vaddq_f32(restemp,atemp);
			}
			s1=vget_low_f32(restemp);
			s2=vget_high_f32(restemp);
			s1=vpadd_f32(s1,s2);
			s1=vpadd_f32(s1,s1);
			vst1_lane_f32(res*n*i+j,s1,0);
//			s1=_mm256_extractf128_ps(restemp,0);
//			s2=_mm256_extractf128_ps(restemp,1);
//			s1=_mm_hadd_ps(s1,s2);
//			s1=_mm_hadd_ps(s1,s1);
//			s1=_mm_hadd_ps(s1,s1);
			//_mm_storeu_ps(res+n*i+j,restemp);
//			restemp=_mm256_hadd_ps(restemp,restemp);
//			restemp=_mm256_hadd_ps(restemp,restemp);
//			restemp=_mm256_hadd_ps(restemp,restemp);
//			_mm_store_ss(res+n*i+j,s1);
		}
	}
}
void cacheNormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
//	__m256 atemp;
//	__m256 btemp;
//	__m256 mulTemp;
//	__m256 resTemp;
//	resTemp=_mm256_setzero_ps(); 
//	for(int i=0;i<n*m;i+=8)
//		_mm256_storeu_ps(res+i,resTemp);

	for(int i=0;i<n;i++)
	{
		for(int k=0;k<s;k++)
		{
			for(int j=0;j<m;j+=8)
			{
				//res[n*i+j]+=(a[n*i+k]*b[s*k+j]);
				//res[n*i+j+1]+=(a[n*i+k]*b[s*k+j+1]);
				//res[n*i+j+2]+=(a[n*i+k]*b[s*k+j+2]);
				//res[n*i+j+3]+=(a[n*i+k]*b[s*k+j+3]);
				//atemp=_mm_set1_ps(a[n*i+k]);

				//atemp=_mm256_load1_ps(a+n*i+k);
//				atemp=_mm256_set1_ps(a[n*i+k]);
//				btemp=_mm256_load_ps(b+s*k+j);
//				mulTemp=_mm256_mul_ps(atemp,btemp);
//				resTemp=_mm256_load_ps(res+n*i+j);
//				resTemp=_mm256_add_ps(resTemp,mulTemp);
//				_mm256_store_ps(res+n*i+j,resTemp);
			}
		}
	}
}
void MatAdd(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
//	__m256 atemp;
//	__m256 btemp;
//	__m256 resTemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=8)
	{
		//res[i*resRowLen+j]=a[i*aRowLen+j]+b[i*bRowLen+j];
		//res[i*resRowLen+j+1]=a[i*aRowLen+j+1]+b[i*bRowLen+j+1];
		//res[i*resRowLen+j+2]=a[i*aRowLen+j+2]+b[i*bRowLen+j+2];
		//res[i*resRowLen+j+3]=a[i*aRowLen+j+3]+b[i*bRowLen+j+3];
//		atemp=_mm256_load_ps(a+i*aRowLen+j);
//		btemp=_mm256_load_ps(b+i*bRowLen+j);
//		resTemp=_mm256_add_ps(atemp,btemp);
//		_mm256_store_ps(res+i*resRowLen+j,resTemp);
	}
}
void MatSub(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
//	__m256 atemp;
//	__m256 btemp;
//	__m256 resTemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=8)
	{
		//res[i*resRowLen+j]=a[i*aRowLen+j]-b[i*bRowLen+j];
		//res[i*resRowLen+j+1]=a[i*aRowLen+j+1]-b[i*bRowLen+j+1];
		//res[i*resRowLen+j+2]=a[i*aRowLen+j+2]-b[i*bRowLen+j+2];
		//res[i*resRowLen+j+3]=a[i*aRowLen+j+3]-b[i*bRowLen+j+3];
//		atemp=_mm256_load_ps(a+i*aRowLen+j);
//		btemp=_mm256_load_ps(b+i*bRowLen+j);
//		resTemp=_mm256_sub_ps(atemp,btemp);
//		_mm256_store_ps(res+i*resRowLen+j,resTemp);
	}
}
void Strassen(int N,
				int aRowLen,const dType*a,
				int bRowLen,const dType*b,
				int resRowLen,dType*res)
{
	if(N<=8)
	{
//		__m256 atemp;
//		__m256 btemp;
//		__m256 multemp;
//		__m256 restemp;
//		restemp=_mm256_setzero_ps();
		for(int i=0;i<N;i++)
		for(int j=0;j<N;j+=8)
		{
//			_mm256_store_ps(res+resRowLen*i+j,restemp);
//			res[resRowLen*i+j]=0;
//			res[resRowLen*i+j+1]=0;
//			res[resRowLen*i+j+2]=0;
//			res[resRowLen*i+j+3]=0;
		}

		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j+=8)
		{
//			res[resRowLen*i+j]+=(a[aRowLen*i+k]*b[bRowLen*k+j]);
//			res[resRowLen*i+j+1]+=(a[aRowLen*i+k]*b[bRowLen*k+j+1]);
//			res[resRowLen*i+j+2]+=(a[aRowLen*i+k]*b[bRowLen*k+j+2]);
//			res[resRowLen*i+j+3]+=(a[aRowLen*i+k]*b[bRowLen*k+j+3]);
			
			//atemp=_mm_load1_ps(a+aRowLen*i+k);
//			atemp=_mm256_set1_ps(a[aRowLen*i+k]);
//			btemp=_mm256_load_ps(b+bRowLen*k+j);
//			multemp=_mm256_mul_ps(atemp,btemp);
//			restemp=_mm256_load_ps(res+resRowLen*i+j);
//			restemp=_mm256_add_ps(restemp,multemp);
//			_mm256_store_ps(res+resRowLen*i+j,restemp);
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
    //dType*a=new dType[N*N];
    dType*a=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    for(int i=0;i<N*N;i++)a[i]=i;
    //dType*b=new dType[N*N];
    dType*b=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    for(int i=0;i<N*N;i++)b[i]=N*N-1-i;
    //dType*c1=new dType[N*N];
    //dType*c2=new dType[N*N];
    //dType*c3=new dType[N*N];
    dType*c1=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c2=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
    dType*c3=(dType*)_mm_malloc(N*N*sizeof(dType),ALIGN_N);
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
    printf("Normal_A512 elapsed %.3f ms\n",tN);
    printf("Cache_A512 elasped %.3f ms\n",tC);
    printf("Strassen_A512 elasped %.3f ms\n",tS);
//    freopen("Log44.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//    if(!(c1[i]==c2[i]&&c2[i]==c3[i]))
//	{
//		printf("Result Inconsistent at %d,%d\n",i/N,i%N);
//		printf("c1=%f c2=%f c3=%f\n",c1[i],c2[i],c3[i]);
//		printf("rate=%f\n\n",(float)c1[i]/c3[i]);
//	}
//
    freopen("cc1_A512.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c1[i]);
	freopen("cc2_A512.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c2[i]);
	freopen("cc3_A512.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c3[i]);
//    delete[]a;
//    delete[]b;
//    delete[]c1;
//    delete[]c2;
//    delete[]c3;
	_mm_free(a);
	_mm_free(b);
	_mm_free(c1);
	_mm_free(c2);
	_mm_free(c3);
    return 0;
}
