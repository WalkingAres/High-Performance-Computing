#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
__global__ void matrix_mul_gpu(float *A, float * B, float * C, int col_a, int col_b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
                
    int sum = 0;
    for(int k=0;k<col_a;k++)
    {
        sum += A[i*col_a+k]*B[k*col_b+j];
    }
    C[i*col_b+j] = sum;
}
 
int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        printf("lack of initial arguments !\n");
        return 0;
    }

    int m,n,k,blocksize;
    // m , n , k
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    blocksize = atoi(argv[4]);


    float *A = (float *)malloc(sizeof(float) * m * n);
    float *B = (float *)malloc(sizeof(float) * n * k);
    float *C = (float *)malloc(sizeof(float) * m * k);
    //malloc device memory
    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, sizeof(float) *m*n);
    cudaMalloc((void**)&dB, sizeof(float) *n*k);
    cudaMalloc((void**)&dC, sizeof(float) *m*k);
    
    for (int i = 0; i < m*n; i++) {
        srand(i);
        A[i] = rand() % 10;
    }

    for (int i = 0; i < n*k; i++) {
        srand(i);
        B[i] = rand() % 10;
    }
         
    
    struct timeval start, end;
    gettimeofday( &start, NULL );

    cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    int blockx = pow(2,(int)(log2(blocksize))/2);
    int blocky;
    if(blockx*blockx == blocksize) blocky = blockx;
    else blocky = 2*blockx;

    dim3 Block(blockx, blocky);
    dim3 Grid((m+Block.x-1)/ Block.x, (k+Block.y-1)/ Block.y );
    
    matrix_mul_gpu <<<Grid, Block >>> (dA, dB, dC, n,k);
    
    //拷贝计算数据-一级数据指针
    cudaMemcpy(C, dC, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
                                                                                            
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("Block(%d,%d)   Grid(%d,%d).\n", Block.x, Block.y, Grid.x, Grid.y);
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

        //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);


    return 0;
}
