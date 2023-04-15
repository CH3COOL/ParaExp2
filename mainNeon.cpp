#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<malloc.h>
#include<sys/time.h>
#include<unistd.h>
#include <arm_neon.h> 
#define ALIGN_N 32
typedef float dType;
void NormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
	float32x4_t atemp,btemp,restemp;
	float32x2_t s1,s2;
	restemp=vdupq_n_f32(0.0);
	for(int i=0;i<n*m;i+=4)
	 	vst1q_f32(res+i,restemp);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			restemp=vdupq_n_f32(0.0);
			for(int k=0;k<s;k+=4)
			{
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
			vst1_lane_f32(res+n*i+j,s1,0);
		}
	}
}
void cacheNormalMul(int n,int s,int m,dType*a,dType*b,dType*res)
{
    //清零
    float32x4_t atemp,btemp,restemp;
    restemp=vdupq_n_f32(0.0);
    for(int i=0;i<n*m;i+=4)
        vst1q_f32(res+i,restemp);
	for(int i=0;i<n;i++)
	{
		for(int k=0;k<s;k++)
		{
			for(int j=0;j<m;j+=4)
			{
				//累加与存储
                atemp=vld1q_f32(a+n*i+k);//load a
                btemp=vld1q_f32(b+s*k+j);//load b
                atemp=vmulq_f32(atemp,btemp);//a=a.*b
                restemp=vld1q_f32(res+n*i+j);//load res from 
                restemp=vaddq_f32(restemp,atemp);//res+=atemp
                vst1q_f32(res+n*i+j,restemp);//store restemp
			}
		}
	}
}
void MatAdd(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
    float32x4_t atemp,btemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=4)
	{
		//load;加法;store
        atemp=vld1q_f32(a+i*aRowLen+j);
        btemp=vld1q_f32(b+i*bRowLen+j);
        atemp=vaddq_f32(atemp,btemp);
        vst1q_f32(res+i*resRowLen+j,atemp);
	}
}
void MatSub(int N,
			int aRowLen,const dType*a,
			int bRowLen,const dType*b,
			int resRowLen,dType*res)
{
    float32x4_t atemp,btemp;
	for(int i=0;i<N;i++)
	for(int j=0;j<N;j+=4)
	{
		//load;减法;store
        atemp=vld1q_f32(a+i*aRowLen+j);
        btemp=vld1q_f32(b+i*bRowLen+j);
        atemp=vsubq_f32(atemp,btemp);
        vst1q_f32(res+i*resRowLen+j,atemp);
	}
}
void Strassen(int N,
				int aRowLen,const dType*a,
				int bRowLen,const dType*b,
				int resRowLen,dType*res)
{
	if(N<=8)
	{
        //reg=0
        float32x4_t atemp,btemp,restemp;
        restemp=vdupq_n_f32(0.0);
		for(int i=0;i<N;i++)
		for(int j=0;j<N;j+=4)
		{
            //清零
            vst1q_f32(res+resRowLen*i+j,restemp);
		}

		for(int i=0;i<N;i++)
		for(int k=0;k<N;k++)
		for(int j=0;j<N;j+=4)
		{
            //乘法、累加、store
            atemp=vld1q_f32(a+aRowLen*i+k);
            btemp=vld1q_f32(b+bRowLen*k+j);
            atemp=vmulq_f32(atemp,btemp);
            restemp=vld1q_f32(res+resRowLen*i+j);
            restemp=vaddq_f32(restemp,atemp);
            vst1q_f32(res+resRowLen*i+j,restemp);
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
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz),
		(dType*)memalign(ALIGN_N,sz)
	};
	dType*T=(dType*)memalign(ALIGN_N,sz);
  	dType*U=(dType*)memalign(ALIGN_N,sz);
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
    int N=16;
    dType*a=(dType*)memalign(ALIGN_N,N*N*sizeof(dType));
    for(int i=0;i<N*N;i++)a[i]=i;
    dType*b=(dType*)memalign(ALIGN_N,N*N*sizeof(dType));
    for(int i=0;i<N*N;i++)b[i]=N*N-1-i;
    dType*c1=(dType*)memalign(ALIGN_N,N*N*sizeof(dType));
    dType*c2=(dType*)memalign(ALIGN_N,N*N*sizeof(dType));
    dType*c3=(dType*)memalign(ALIGN_N,N*N*sizeof(dType));
    struct timeval beginTime,EndTime;

    float tN,tC,tS;

    gettimeofday(&beginTime,NULL);
    NormalMul(N,N,N,a,b,c1);
    gettimeofday(&EndTime,NULL);
	tN=(EndTime.tv_sec-beginTime.tv_sec)*1000.0+(EndTime.tv_usec-beginTime.tv_usec)/1000.0;
	
	gettimeofday(&beginTime,NULL);
    cacheNormalMul(N,N,N,a,b,c2);
    gettimeofday(&EndTime,NULL);
    tC=(EndTime.tv_sec-beginTime.tv_sec)*1000.0+(EndTime.tv_usec-beginTime.tv_usec)/1000.0;
    
    gettimeofday(&beginTime,NULL);
    Strassen(N,N,a,N,b,N,c3);
    gettimeofday(&EndTime,NULL);
    tS=(EndTime.tv_sec-beginTime.tv_sec)*1000.0+(EndTime.tv_usec-beginTime.tv_usec)/1000.0;
    
    printf("N=%d\n",N);
    printf("Normal_Neon elapsed %.3f ms\n",tN);
    printf("Cache_Neon elasped %.3f ms\n",tC);
    printf("Strassen_Neon elasped %.3f ms\n",tS);
//    freopen("Log44.txt","w",stdout);
//    for(int i=0;i<N*N;i++)
//    if(!(c1[i]==c2[i]&&c2[i]==c3[i]))
//	{
//		printf("Result Inconsistent at %d,%d\n",i/N,i%N);
//		printf("c1=%f c2=%f c3=%f\n",c1[i],c2[i],c3[i]);
//		printf("rate=%f\n\n",(float)c1[i]/c3[i]);
//	}
//
    freopen("cc1_Neon.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c1[i]);
	freopen("cc2_Neon.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c2[i]);
	freopen("cc3_Neon.txt","w",stdout);
    for(int i=0;i<N*N;i++)
		printf("%f\n",c3[i]);
	free(a);
	free(b);
	free(c1);
	free(c2);
	free(c3);
    return 0;
}
