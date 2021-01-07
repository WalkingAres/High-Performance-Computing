#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc < 3)
    {
        printf("lack of initial arguments !\n");
        return 0;
    }

    int m,n,k;
    // m , n , k 
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

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

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, sizeof(float) *m*n);
    cudaMalloc((void**)&dB, sizeof(float) *n*k);
    cudaMalloc((void**)&dC, sizeof(float) *m*k);

    struct timeval start, end;
    gettimeofday( &start, NULL );

    float alpha = 1;
    float beta = 0;

    cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_N,
                k,m,n,
                &alpha,
                dB,k,
                dA,n,
                &beta,
                 dC,k);
    cudaMemcpy(C, dC, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/(float)1000);

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