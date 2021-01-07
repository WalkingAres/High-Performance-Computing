#include <mpi.h>
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <stdio.h>
#include<iostream>
using namespace std;
#define M 500
#define N 500
#define master 0
double diff;
double epsilon = 0.001;
int i;
int iterations;
int iterations_print;
int j;
double mean;
double my_mean;
double my_diff;
double u[M*N];
double w[M*N];
double buffer[M*N];
double wtime;
bool check()
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			if (w[i*N + j] != u[i*N + j])
				return false;
	return true;
}
int main(int argc, char** argv) {
	// ????? MPI ????
	MPI_Init(NULL, NULL);
	// ??????????¡¤???????????§á???????????????
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// ?????????????
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	mean = 0;
#pragma omp parallel shared ( w ) private ( i, j )
	{
#pragma omp for
		for (i = 1; i < M - 1; i++)
		{
			w[i*N + 0] = 100.0;
		}
#pragma omp for
		for (i = 1; i < M - 1; i++)
		{
			w[i*N + N - 1] = 100.0;
		}
#pragma omp for
		for (j = 0; j < N; j++)
		{
			w[(M - 1)*N + j] = 100.0;
		}
#pragma omp for
		for (j = 0; j < N; j++)
		{
			w[j] = 0.0;
		}
	}
	//cout << w[(M - 1)*M] << endl;
	if (rank == master)
	{
	
		printf("\n");	
		printf("HEATED_PLATE_OPENMP\n");
		printf("  C/OpenMP version\n");
		printf("  A program to solve for the steady state temperature distribution\n");
		printf("  over a rectangular plate.\n");
		printf("\n");
		printf("  Spatial grid of %d by %d points.\n", M, N);
		printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
		printf("  Number of threads =          4\n");
		int block1 = (M - 2) / 4;
		for (int i = 1; i < block1 + 1; i++)
			my_mean = my_mean + w[i*N] + w[i*N+N - 1];
		int block2 = N / 4;
		for (int j = 0; j < block2; j++)
			my_mean = my_mean + w[(M - 1)*N + j] + w[j];
		//cout << my_mean << endl;
	}
	else
	{	
		int block1 = (M - 2) / 4;
		int block2 = N / 4;
		if (rank != 3)
		{
			for (int i = 1 + block1 * rank; i < 1 + block1 * (rank + 1); i++)
			{
				my_mean = my_mean + w[i*N] + w[i*N + N - 1];
			}
			for (int j = rank*block2; j < block2*(rank+1); j++)
				my_mean = my_mean + w[(M - 1)*N + j] + w[j];
		}
		else
		{
			for(int i = 1+3*block1;i<M-1;i++)
				my_mean = my_mean + w[i*N] + w[i*N + N - 1];
			for (int j = 3 * block2; j < N; j++)
				my_mean = my_mean + w[(M - 1)*N + j] + w[j];
		}
		cout << my_mean << endl;
		cout << "shit111" << endl;;
	}
	
	
	
	MPI_Reduce(&my_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);
	MPI_Reduce(&my_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (rank == master)
	{
		mean = mean / (double)(2 * M + 2 * N - 4);
		cout << endl;
		cout << "mean =  " << mean << endl;
		cout << endl << " Iteration  Change" << endl;
		cout << endl;
	}
	MPI_Bcast(&mean,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
		#pragma omp parallel shared ( mean, w ) private ( i, j )
		{
#pragma omp for
			for (i = 1; i < M - 1; i++)
			{
				for (j = 1; j < N - 1; j++)
				{
					w[i*N + j] = mean;
				}
			}
		}
		iterations = 0;
		iterations_print = 1;
		wtime = MPI_Wtime();
		diff = epsilon;

		while (epsilon <= diff)
		{
			# pragma omp for
			for (i = 0; i < M; i++)
			{
				for (j = 0; j < N; j++)
				{
					u[i*N + j] = w[i*N + j];
				}
			}
			int block = (M-2) / 4;
			my_diff = 0;
			if (rank == master)
			{
				for(int i=1+3*block;i<M-1;i++)
					for(int j=1;j<N-1;j++)
						w[i*N+j] = (u[(i - 1)*N+j] + u[(i + 1)*N + j] + u[i*N + j - 1] + u[i*N + j + 1]) / 4.0;
				for (int i = 1 + 3 * block; i < M - 1; i++)
					for (int j = 1; j < N - 1; j++)
						if (my_diff < fabs(w[i*N + j] - u[i*N + j]))
							my_diff = fabs(w[i*N + j] - u[i*N + j]);
				//cout << "master: " << my_diff << endl;
				for (int label = 1; label <= 3; label++)
				{
					MPI_Recv(buffer, M*N, MPI_DOUBLE, label, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					for (int i = 1 + (label - 1)*block; i < 1 + label * block; i++)
						for (int j = 1; j < N - 1; j++)
							w[i*N + j] = buffer[i*N + j];
				}
				//cout << rank <<"  "<< my_diff << endl;
			}
			else
			{
				int pos = 0;
				for (int i = 1 + (rank - 1)*block; i < 1 + rank * block; i++)
					for (int j = 1; j < N - 1; j++)
						w[i*N + j] = (u[(i - 1)*N + j] + u[(i + 1)*N + j] + u[i*N + j - 1] + u[i*N + j + 1]) / 4.0;
				pos = 0;
				for (int i = 1 + (rank - 1)*block; i < 1 + rank * block; i++)
					for (int j = 1; j < N - 1; j++)
					{
						if (my_diff < fabs(w[i*N + j] - u[i*N + j]))
							my_diff = fabs(w[i*N + j] - u[i*N + j]);
					}
				//cout << "rank " << rank << " " << my_diff << endl;
				MPI_Send(w, N*M, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
				//cout << rank << "  " << my_diff << endl;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
			MPI_Bcast(w, M*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (rank == master)
			{
				iterations++;
				if (iterations == iterations_print)
				{
					cout << " " << iterations <<"   "<< diff << endl;
					iterations_print = 2 * iterations_print;
				}
			}
		}
	// ???????????????
		if (rank == master)
		{
			wtime = MPI_Wtime(); - wtime;

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

		}
	// ??? MPI ???§»???
	MPI_Finalize();
}