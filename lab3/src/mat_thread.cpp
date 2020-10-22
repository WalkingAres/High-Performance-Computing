#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <sys/time.h>

/**  m , n , k , num_threads(运行线程数量)
***  A , B , C  共享数据（矩阵数据）
*** col_count 每个线程得到的任务数量  
*** */

int m, n, k, num_threads;
int *A, *B, *C;
int col_count;

void *matrix_mul_vec(void *rank)
{
    // first 任务开始的列向量索引
    // last  任务结束的列向量索引
    int myrank = *(int *)rank;
    int first, last;
    first = col_count * myrank;
    last = col_count + first;

    for (int i = 0; i < m; i++)
    {
        for (int j = first; j < last; j++)
        {
            int sum = 0;
            for (int l = 0; l < n; l++)
            {
                sum += A[i * n + l] * B[j * n + l];
            }
            C[j * m + i] = sum;
        }
    }

    return NULL;
}

int main(int argc, char **arg)
{
    if (argc < 4)
    {
        printf("lack of initial arguments !\n");
        return 0;
    }


    // m , n , k , num_threads(运行线程数量) 初始化
    m = atoi(arg[1]);
    n = atoi(arg[2]);
    k = atoi(arg[3]);
    num_threads = atoi(arg[4]);

    //随机矩阵初始化
    A = new int[m * n];
    // B矩阵 以列形式存储 即B的转置存储 C矩阵同理
    B = new int[k * n];
    C = new int[m * k];

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

    
    col_count = k / num_threads;            // 根据线程数量划分任务
    pthread_t *thread;
    int* rank;                              //定义线程序号
    rank = new int[num_threads];
    thread = new pthread_t[num_threads];    //子线程

    struct timeval start, end;
    gettimeofday( &start, NULL );

    for (int i = 0; i < num_threads; i++)
    {   
        // 创建子线程
        rank[i] = i;
        pthread_create(thread + i, NULL, matrix_mul_vec, rank+i);
    }

    for (int i = 0; i < num_threads; i++)
    {
        // 等待子线程运行结束
        pthread_join(thread[i], NULL);
    }

    gettimeofday( &end, NULL );

    double  timeuse = (1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec)/(double)1000; 
    printf("m n k : %d %d %d\ntime: %lf ms\n", m, n, k, timeuse);

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
            fprintf(c, "%d ", C[j * m + i]);
        fprintf(c, "\n");
    }
    fprintf(c, "];");

    fclose(a);
    fclose(b);
    fclose(c);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] rank;
    delete[] thread;

    return 0;
}