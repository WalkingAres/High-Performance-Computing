#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include "parallelfor.h"

#define M 250
#define N 250

double u[M][N];
double w[M][N];

int num_threads = 4;

// 表示行向量/列向量
typedef enum vecType
{
    col,   
    row
} vecType;

// 实现矩阵行向量、列向量 赋值
// 给type类型的向量的（index为向量在矩阵中的索引）赋值为value
typedef struct arg1
{
    vecType type;
    int index;
    double value;
} arg1;

void *assignVec(void *arg)
{
    // 参数解析
    targ *parg1 = (targ *)arg;
    arg1 *parg2 = (arg1 *)(parg1->arg);
    // 判断向量类型 行/列
    if (parg2->type == col)
    {
        for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        {
            // 向量赋值
            w[i][parg2->index] = parg2->value;
        }
    }
    else
    {
        // 向量赋值
        for (int i = parg1->first; i < parg1->last; i += parg1->stride)
            w[parg2->index][i] = parg2->value;
    }
    return NULL;
}

// reduction +
typedef struct arg2
{
    vecType type;
    int index1;
    int index2;
    double *result;
} arg2;

pthread_mutex_t mutex_sum = PTHREAD_MUTEX_INITIALIZER;
void *reduction(void *arg)
{
    targ *parg1 = (targ *)arg;
    arg2 *parg2 = (arg2 *)(parg1->arg);
    double local_result = 0.0;
    if (parg2->type == col)
    {
        for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        {
            pthread_mutex_lock(&mutex_sum);
            *(parg2->result) = *(parg2->result) + w[i][parg2->index1] + w[i][parg2->index2];
            pthread_mutex_unlock(&mutex_sum);
        }
    }
    else
    {
        for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        {
            pthread_mutex_lock(&mutex_sum);
            *(parg2->result) = *(parg2->result) + w[parg2->index1][i] + w[parg2->index2][i];
            pthread_mutex_unlock(&mutex_sum);
        }
    }
}

// initialize w[i][j] = mean

void *set_w(void *arg)
{
    targ *parg1 = (targ *)arg;
    double *mean = (double *)(parg1->arg);
    for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        for (int j = 1; j < N - 1; j++)
            w[i][j] = *mean;
}

// save old in u
void *save2u(void *arg)
{
    targ *parg1 = (targ *)arg;
    for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        for (int j = 0; j < N; j++)
            u[i][j] = w[i][j];
    return NULL;
}

// new estimate
void *new_w(void *arg)
{
    targ *parg1 = (targ *)arg;
    for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        for (int j = 1; j < N - 1; j++)
            w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;

    return NULL;
}

// diff

pthread_mutex_t mutex_diff = PTHREAD_MUTEX_INITIALIZER;

void *compute_diff(void *arg)
{
    targ *parg1 = (targ *)arg;
    double *diff = (double *)(parg1->arg);
    double local_diff = 0.0;
    // 局部diff
    for (int i = parg1->first; i < parg1->last; i += parg1->stride)
        for (int j = 1; j < N - 1; j++)
            if (local_diff < fabs(w[i][j] - u[i][j]))
            {
                local_diff = fabs(w[i][j] - u[i][j]);
            }
    // 互斥访问
    pthread_mutex_lock(&mutex_diff);
    if (*diff < local_diff)
        *diff = local_diff;
    pthread_mutex_unlock(&mutex_diff);
    return NULL;
}

int main(int argc, char *argv[])
{
    double diff;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;
    double my_diff;

    double wtime;

    printf("\n");
    printf("HEATED_PLATE_OPENMP\n");
    printf("  C/OpenMP version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processors available = %d\n", omp_get_num_procs());
    printf("  Number of threads =              %d\n", omp_get_max_threads());
    /*
  Set the boundary values, which don't change. 
*/
    mean = 0.0;


    arg1 *arg = malloc(sizeof(arg1));

    arg->type = col;
    arg->index = 0;
    arg->value = 100.0;
    parallel_for(1, M - 1, 1, assignVec, arg, num_threads);

    arg->index = N - 1;
    parallel_for(1, M - 1, 1, assignVec, arg, num_threads);

    arg->type = row;
    arg->index = M - 1;
    parallel_for(0, N, 1, assignVec, arg, num_threads);

    arg->index = 0;
    arg->value = 0.0;
    parallel_for(0, N, 1, assignVec, arg, num_threads);

    arg2 *arg_reduction = malloc(sizeof(arg2));

    arg_reduction->type = col;
    arg_reduction->index1 = 0;
    arg_reduction->index2 = N - 1;
    arg_reduction->result = &mean;
    parallel_for(1, M - 1, 1, reduction, arg_reduction, num_threads);

    arg_reduction->type = row;
    arg_reduction->index2 = M - 1;
    parallel_for(0, N, 1, reduction, arg_reduction, num_threads);
    /*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);
    /* 
  Initialize the interior solution to the mean value.
*/
    parallel_for(1, M - 1, 1, set_w, &mean, num_threads);
    /*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    wtime = omp_get_wtime();

    diff = epsilon;

    while (epsilon <= diff)
    {
        parallel_for(0, M, 1, save2u, NULL, num_threads);
        parallel_for(1, M - 1, 1, new_w, NULL, num_threads);
        /*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
        diff = 0.0;

        parallel_for(1, M - 1, 1, compute_diff, &diff, num_threads);

        if (diff < my_diff)
            diff = my_diff;

        iterations++;
        if (iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    wtime = omp_get_wtime() - wtime;

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    /*
  Terminate.
*/
    printf("\n");
    printf("HEATED_PLATE_OPENMP:\n");
    printf("  Normal end of execution.\n");

    return 0;
}
