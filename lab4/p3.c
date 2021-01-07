#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include "parallelfor.h"

int *A, *B, *C;
int m, n, k;

struct arg_index
{
    int first;
    int last;
    int stride;
};              


void * mat_mul(void * index) {
    struct arg_index *pindex = (struct arg_index *) index;
    for(int i=pindex->first; i < pindex->last; i += pindex->stride){
        for(int j = 0; j < k; j++) {
            C[i*n+j] = 0;
            for(int l = 0; l < n; l++) {
                C[i*n+j] += A[i*n+j]*B[j*n+l];
            }
        }
    }
    return NULL;
}

// void parallel_for(int start, int end, int stride, void * (*function)(void *), void * arg, int num_threads) {
//     // 平均分配任务
//     // 总任务量
//     int total = (end - start + 1) / stride;
//     int q = total /  num_threads;
//     int r = total %  num_threads;
//     // 每个线程的任务
//     int count;

//     pthread_t * threads = malloc(sizeof(pthread_t)*num_threads);
//     struct arg_index * parg = malloc(sizeof(struct arg_index)*num_threads);
//     for(int i=0; i<num_threads; i++) {
//         // 初始化线程入口参数 （任务分配）
//         if(i < r) {
//             count = q + 1;
//             parg[i].first = i*count;
//         }
//         else {
//             count = q;
//             parg[i].first = rank*count + r;
//         }
//         parg[i].last = parg[i].first + count;
//         parg[i].stride = stride;
//         // 创建子线程
//         pthread_create(threads+i,NULL,function,parg+i);
//     }

//     for (int i = 0; i < num_threads; i++){
//         // 等待子线程运行结束
//         pthread_join(threads[i], NULL);
//     }

//     free(threads);
//     free(parg);

// }


int main(int argc, char *argv[] ) {
    if(argc < 4) printf("please input m, n, k, numbers of threads!\n");
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    // C = A*B
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

// #pragma omp parallel for num_threads(num_threads)
//     for(int i = 0; i < m; i++) {
//         for(int j = 0; j < k; j++) {
//             C[i*k+j] = 0;
//             for(int l = 0; l < n; l++) {
//                 C[i*k+j] += A[i*n+l]*B[j*n+l];
//             }
//         }
//     } 

    parallel_for(0,m,1,mat_mul,NULL,num_threads);

    gettimeofday( &end, NULL );

    double  timeuse = (1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec)/(double)1000; 
    printf("m n k number of threads: %d %d %d %d\ntime: %lf ms\n", m, n, k, num_threads,timeuse);

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