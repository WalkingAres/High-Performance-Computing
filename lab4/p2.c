#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[] ) {
    if(argc < 4) printf("please input m, n, k, numbers of threads!\n");
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    // C = A*B
    int *A, *B, *C;
    A = malloc(m*n*sizeof(int));
    B = malloc(n*k*sizeof(int));
    C = malloc(m*k*sizeof(int));

    // 初始化矩阵A，B
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            srand(i + j);
            A[i * n + j] = rand() % 10;
        }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
        {
            srand(i + j + 123);
            B[j * n + i] = rand() % 10;
        }

    struct timeval start, end;
    gettimeofday( &start, NULL );

#pragma omp parallel for num_threads(num_threads) \
    schedule(dynamic,8)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            C[i*k+j] = 0;
            for(int l = 0; l < n; l++) {
                C[i*k+j] += A[i*n+l]*B[j*n+l];
            }
        }
    } 

    gettimeofday( &end, NULL );

    double  timeuse = (1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec)/(double)1000; 
    printf("m n k number of threads: %d %d %d %d\ntime: %lf s\n", m, n, k, num_threads,timeuse/(double)1000);

    FILE *a, *b, *c;
    a = fopen("matrixA", "wb");
    b = fopen("matrixB", "wb");
    c = fopen("matrixC", "wb");

    fprintf(a, "A = [ \n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(a, "%d ", A[i * n + j]);
        fprintf(a, "\n");
    }
    fprintf(a, "];");

    fprintf(b, "B = [ \n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
            fprintf(b, "%d ", B[j * n + i]);
        fprintf(b, "\n");
    }
    fprintf(b, "];");

    fprintf(c, "C = [ \n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            fprintf(c, "%d ", C[i * k + j]);
        fprintf(c, "\n");
    }
    fprintf(c, "];");

    return 0;
}