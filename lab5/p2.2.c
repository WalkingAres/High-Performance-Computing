#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define M 500
#define N 500

double u[M * N];
double w[M * N];
double z[M * N];

typedef struct index
{
    int first;
    int last;
} index;

index get_task(int start, int end, int rank, int num_threads)
{
    int total = end - start;
    int q = total / num_threads;
    int r = total % num_threads;
    // 每个线程的任务
    int count;

    index my_index;
    if (rank < r)
    {
        count = q + 1;
        my_index.first = start + rank * count;
    }
    else
    {
        count = q;
        my_index.first = start + rank * count;
    }
    my_index.last = my_index.first + count;
    return my_index;
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

    int comm_sz;
    int my_rank;
    int num_threads = 6;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0)
    {

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

#pragma omp parallel shared(w) private(i, j)
        {
#pragma omp for
            for (i = 1; i < M - 1; i++)
            {
                w[i * N + 0] = 100.0;
            }
#pragma omp for
            for (i = 1; i < M - 1; i++)
            {
                w[i * N + N - 1] = 100.0;
            }
#pragma omp for
            for (j = 0; j < N; j++)
            {
                w[(M - 1) * N + j] = 100.0;
            }
#pragma omp for
            for (j = 0; j < N; j++)
            {
                w[j] = 0.0;
            }
        }
    }
    MPI_Bcast(w, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mean = 0.0;

#pragma omp parallel shared(w) private(i, j)
    {
#pragma omp for reduction(+ \
                          : mean)
        for (i = 1; i < M - 1; i++)
        {
            mean = mean + w[i * N + 0] + w[i * N + N - 1];
        }
#pragma omp for reduction(+ \
                          : mean)
        for (j = 0; j < N; j++)
        {
            mean = mean + w[(M - 1) * N + j] + w[j];
        }
    }

    mean = mean / (double)(2 * M + 2 * N - 4);

    if (my_rank == 0)
    {
        printf("\n");
        printf("  MEAN = %f\n", mean);
    }

    index my_index = get_task(1, M - 1, my_rank, num_threads);
    for (int i = my_index.first; i < my_index.last; i++)
    {
        for (int j = 0; j < N - 1; j++)
        {
            w[i * N + j] = mean;
        }
    }
    MPI_Bcast(w + my_index.first * N, my_index.last - my_index.first, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    iterations = 0;
    iterations_print = 1;
    if (my_rank == 0)
    {
        printf("\n");
        printf(" Iteration  Change\n");
        printf("\n");
        wtime = omp_get_wtime();
    }

    diff = epsilon;
    while (epsilon <= diff)
    {
        index my_index = get_task(0, M, my_rank, num_threads);
        for (int i = my_index.first; i < my_index.last; i++)
        {
            for (int j = 0; j < N; j++)
            {
                u[i * N + j] = w[i * N + j];
            }
        }
        MPI_Bcast(u + my_index.first * N, my_index.last - my_index.first, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        my_index = get_task(1, M - 1, my_rank, num_threads);
        for (i = my_index.first; i < my_index.last; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                w[i * N + j] = (u[(i - 1) * N + j] + u[(i + 1) * +j] + u[i * N + j - 1] + u[i * N + j + 1]) / 4.0;
            }
        }
        MPI_Bcast(w + my_index.first * N, my_index.last - my_index.first, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        diff = 0.0;
        if(my_rank == 0)  {
#pragma omp parallel shared(diff, u, w) private(i, j, my_diff)
        {
            my_diff = 0.0;
#pragma omp for
            for (i = 1; i < M - 1; i++)
            {
                for (j = 1; j < N - 1; j++)
                {
                    if (my_diff < fabs(w[i*N+j] - u[i*N+j]))
                    {
                        my_diff = fabs(w[i*N+j] - u[i*N+j]);
                    }
                }
            }
#pragma omp critical
            {
                if (diff < my_diff)
                {
                    diff = my_diff;
                }
            }
        }
        }
        if (my_rank == 0)
        {
            iterations++;
            if (iterations == iterations_print)
            {
                printf("  %8d  %f\n", iterations, diff);
                iterations_print = 2 * iterations_print;
            }
        }
    }
    if (my_rank == 0)
    {
        wtime = omp_get_wtime() - wtime;

        printf("\n");
        printf("  %8d  %f\n", iterations, diff);
        printf("\n");
        printf("  Error tolerance achieved.\n");
        printf("  Wallclock time = %f\n", wtime);

        printf("\n");
        printf("HEATED_PLATE_OPENMP:\n");
        printf("  Normal end of execution.\n");
    }

    MPI_Finalize();
    return 0;
}