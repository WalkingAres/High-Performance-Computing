#include<cstdlib>
#include<cstdio>
#include"mpi.h"
#include<iostream>


struct Task
{
    int first;
    int last;
    int count;
};

typedef Task task;

task get_task(int total, int size, int rank) {
    int q = total / size;
    int r = total % size;
    int count;
    task mytask;
    if(rank < r) {
        count = q + 1;
        mytask.first = rank*count;
    }
    else {
        count = q;
        mytask.first = rank*count + r;
    }
    mytask.last = mytask.first + count;
    mytask.count = count;
    return mytask;
}


int main(int argc, char** argv){
    // 矩阵规模
    int m, n, k;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    double start, end;

    int my_rank, comm_sz, dest;
    task col_task;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    int *A = new int [m*n];
    int *B, *C;

    if(my_rank == 0) {
        //   初始化
        start = MPI_Wtime();
        // B矩阵 以列形式存储 即B的转置存储 C矩阵同理
        B = new int[k*n];
        C = new int[m*k];

        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                srand(i+j);
                A[i*n+j] = rand()%10;
            }

        for(int i=0;i<n;i++)
            for(int j=0;j<k;j++){
                srand(i+j+123);
                B[j*n+i] = rand()%10;
            }
    }

    int count = n*k/comm_sz;
    int col_num = k/comm_sz;
    int *subCol = new int[count];
    int *result = new int[m*col_num]; 
    //分发数据
    MPI_Bcast(A,m*n,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(B,count,MPI_INT,subCol,count,MPI_INT,0,MPI_COMM_WORLD);
    
    for(int i=0;i<m;i++){
        for(int j=0;j<col_num;j++) {
            int sum = 0;
            for(int l=0;l<n;l++) 
                sum += A[i*n+l]*subCol[j*n+l];
            result[j*m+i] = sum;
        }
    }

    MPI_Gather(result,m*col_num,MPI_INT,C,m*col_num,MPI_INT,0,MPI_COMM_WORLD);

    FILE* a, *b, *c;
    a = fopen("matrixA","wb");
    b = fopen("matrixB","wb");
    c = fopen("matrixC","wb");

    if(my_rank == 0) {
        end = MPI_Wtime();

        // printf("A:\n");
        // for(int i=0; i<m;i++) {
        //     for(int j=0;j<n;j++) printf("%d ", A[i*n+j]);
        //     printf("\n");
        // }
        fprintf(a,"A = [ \n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<n;j++) fprintf(a,"%d ",A[i*n+j]);
            fprintf(a,"\n");
        }
        fprintf(a,"];");
        // printf("B:\n");
        // for(int i=0; i<n;i++) {
        //     for(int j=0;j<k;j++) printf("%d ", B[j*n+i]);
        //     printf("\n");
        // }
        fprintf(b,"B = [ \n");
        for(int i=0; i<n;i++) {
            for(int j=0;j<k;j++) fprintf(b,"%d ",B[j*n+i]);
            fprintf(b,"\n");
        }
        fprintf(b,"];");
        // printf("C:\n");
        // for(int i=0; i<m;i++) {
        //     for(int j=0;j<k;j++) printf("%d ", C[j*m+i]);
        //     printf("\n");
        // }
        fprintf(c,"C = [ \n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<k;j++) fprintf(c,"%d ",C[j*m+i]);
            fprintf(c,"\n");
        }
        fprintf(c,"];");

        fclose(a);
        fclose(b);
        fclose(c);

        delete [] A;
        delete [] B;
        delete [] C;

        printf("m n k :%d %d %d\n", m, n, k);
        printf("time:%4lf s \n", end-start);
    }

    delete [] subCol;
    delete [] result;

    MPI_Finalize();

    return 0;
}