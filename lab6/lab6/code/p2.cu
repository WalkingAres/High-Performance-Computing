#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

__global__ void matrix_mul(float *A, float *B, float *C, int col_a, int col_b) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int sum = 0;
    for(int k = 0; k < col_a; k++) {
        sum += A[i*col_a + k]*B[k*col_b+j];
    }
    C[i*col_b+j] = sum;
}

void cudaMul(float *A, float *dB, float *C, int m, int n, int k, int rank, int num_thread, int blocksize){
    int q = m/num_thread;
    int r = m%num_thread;
    int first,count;
    count = q;
    if(rank < r) {
        count = q + 1;
        first = count * rank;
    }
    else {
        first = count * rank + r;
    }


    float * dA,*dC;
    cudaMalloc((void**)&dA, sizeof(int)*count*n);
    cudaMalloc((void**)&dC,sizeof(float)*count*k);

    cudaMemcpy(dA,A+first*n,sizeof(float)*count*n,cudaMemcpyHostToDevice);
    
    int blockx = pow(2,(int)(log2(blocksize))/2);
    int blocky;
    if(blockx*blockx == blocksize) blocky = blockx;
    else blocky = 2*blockx;

    dim3 Block(blockx, blocky);
    dim3 grid((count+block.x-1)/block.x,(k+block.y-1)/block.y);
    
    
    matrix_mul<<<grid,block>>>(dA,dB,dC,n,k);

    cudaMemcpy(C+first*k,dC,sizeof(float)*count*k,cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dC);

}

int main(int argc , char *argv[]) {

    if (argc < 5)
    {
        printf("lack of initial arguments !\n");
        return 0;
    }

    int m,n,k,blocksize,num_thread;
    // m , n , k , num_thread(运行线程数量) 初始化
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    blocksize = atoi(argv[4]);
    num_thread = atoi(argv[5]);

    float *A = (float *)malloc(sizeof(float) * m * n);
    float *B = (float *)malloc(sizeof(float) * n * k);
    float *C = (float *)malloc(sizeof(float) * m * k);

    for (int i = 0; i < m*n; i++) {
        srand(i);
        A[i] = rand() % 10;
    }

    for (int i = 0; i < n*k; i++) {
        srand(i);
        B[i] = rand() % 10;
    }

    float *dB;
    cudaMalloc((void**)&dB, sizeof(float)*n*k);
    
    struct timeval start, end;
    gettimeofday( &start, NULL );
    
    cudaMemcpy(dB,B,sizeof(float)*n*k,cudaMemcpyHostToDevice);
    

    #pragma omp parallel num_threads(num_thread)
    {
        cudaMul(A,dB,C,m,n,k,omp_get_thread_num(),num_thread,blocksize);
    }

    cudaFree(dB);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("block (%d,%d)\nnumber of threads %d\n",blocksize,blocksize,num_thread);
    printf("total time is %f ms\n", timeuse/(float)1000);

    FILE *a, *b, *c;
    a = fopen("matrixA.m", "wb");
    b = fopen("matrixB.m", "wb");
    c = fopen("matrixC.m", "wb");

    fprintf(a, "A = [ \n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(a, "%f ", A[i * n + j]);
        fprintf(a, "\n");
    }
    fprintf(a, "];");

    fprintf(b, "B = [ \n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
            fprintf(b, "%f ", B[i * k + j]);
        fprintf(b, "\n");
    }
    fprintf(b, "];");

    fprintf(c, "C = [ \n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            fprintf(c, "%f ", C[i * k + j]);
        fprintf(c, "\n");
    }
    fprintf(c, "];");


    free(A);
    free(B);
    free(C);

    return 0;
}