#include<cstdlib>
#include<cstdio>
#include"mpi.h"


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


    if(my_rank == 0) {
        //   初始化
        start = MPI_Wtime();
        int * A, *B, *C;
        A = new int[m*n];
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
        
        //分发数据
        for(int i=1;i<comm_sz;i++) { // i = rank
            col_task = get_task(k,comm_sz,i);
            MPI_Send(A,m*n,MPI_INT,i,i,MPI_COMM_WORLD);
            MPI_Send(B+col_task.first*n,col_task.count*n,MPI_INT,i,i,MPI_COMM_WORLD); 
        }

        // 计算
        col_task = get_task(k,comm_sz,my_rank);
        for(int i=0;i<m;i++)
            for(int j=col_task.first;j<col_task.last;j++){
                int sum = 0;
                for(int l=0;l<n;l++) sum += A[i*n+l]*B[j*n+l];
                C[j*m+i] = sum;
            }
        
        // 汇总结果
        for(int i=1;i<comm_sz;i++){
            col_task = get_task(k,comm_sz,i);
            MPI_Recv(C+col_task.first*m,col_task.count*n,MPI_INT,i,i,MPI_COMM_WORLD,NULL);
        }

        end = MPI_Wtime();

        printf("A:\n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<n;j++) printf("%d ", A[i*n+j]);
            printf("\n");
        }
        printf("B:\n");
        for(int i=0; i<n;i++) {
            for(int j=0;j<k;j++) printf("%d ", B[j*n+i]);
            printf("\n");
        }
        printf("C:\n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<k;j++) printf("%d ", C[j*m+i]);
            printf("\n");
        }

        delete [] A;
        delete [] B;
        delete [] C;

        printf("m n k :%d %d %d\n", m, n, k);
        printf("time:%4lf s \n", end-start);
    }
    else {
        col_task = get_task(k,comm_sz,my_rank);
        int *A, * subCol, *result;
        A = new int[m*n];
        subCol = new int[col_task.count*n];
        result = new int[col_task.count*m];

        // 接受数据
        // 矩阵
        MPI_Recv(A,m*n,MPI_INT,0,my_rank,MPI_COMM_WORLD,NULL);
        // 列
        MPI_Recv(subCol,col_task.count*n,MPI_INT,0,my_rank,MPI_COMM_WORLD,NULL);

        // 计算
        for(int i=0;i<m;i++){           
            for(int j=0;j<col_task.count;j++){
                int sum = 0;
                for(int l=0;l<n;l++){
                    sum += A[i*n+l]*subCol[j*n+l];
               }
               result[j*m+i] = sum;
            }
        }


        // 发送结果
        MPI_Send(result,col_task.count*m,MPI_INT,0,my_rank,MPI_COMM_WORLD);

        // 资源回收
    
        delete [] A;
        delete [] subCol;
        delete [] result;
    }

    MPI_Finalize();

    return 0;
}