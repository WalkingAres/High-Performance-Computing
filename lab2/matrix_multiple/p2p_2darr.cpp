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
        int ** A, **B, **C;
        A = new int*[m];
        for(int i=0;i<m;i++) A[i] = new int[n];
        // B矩阵 以列形式存储 B[i][j]表示第i列 第j行的元素
        B = new int*[k];
        for(int i=0;i<k;i++) B[i] = new int[n];
        C = new int*[m];
        for(int i=0;i<m;i++) C[i] = new int[k];
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                srand(i+j);
                A[i][j] = rand()%10;
            }

        for(int i=0;i<n;i++)
            for(int j=0;j<k;j++){
                srand(i+j+123);
                B[j][i] = rand()%10;
            }
        
        //分发数据
        for(int i=1;i<comm_sz;i++) { // i = rank
            col_task = get_task(k,comm_sz,i);
            for(int j=0;j<m;j++)
                MPI_Send(A[j],n,MPI_INT,i,i,MPI_COMM_WORLD);
            for(int j=col_task.first;j<col_task.last;j++)
                MPI_Send(B[j],n,MPI_INT,i,i,MPI_COMM_WORLD); 
        }

        // 计算
        col_task = get_task(k,comm_sz,my_rank);
        for(int i=0;i<m;i++)
            for(int j=col_task.first;j<col_task.last;j++){
                int sum = 0;
                for(int l=0;l<n;l++) sum += A[i][l]*B[j][l];
                C[i][j] = sum;
            }
        
        // 汇总结果
        for(int i=1;i<comm_sz;i++){
            col_task = get_task(k,comm_sz,i);
            for(int j=0;j<m;j++){
                MPI_Recv(&C[j][col_task.first],col_task.count,MPI_INT,i,i,MPI_COMM_WORLD,NULL);
            }
        }

        end = MPI_Wtime();

        printf("A:\n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<n;j++) printf("%d ", A[i][j]);
            printf("\n");
        }
        printf("B:\n");
        for(int i=0; i<n;i++) {
            for(int j=0;j<k;j++) printf("%d ", B[j][i]);
            printf("\n");
        }
        printf("C:\n");
        for(int i=0; i<m;i++) {
            for(int j=0;j<k;j++) printf("%d ", C[i][j]);
            printf("\n");
        }

        for(int i=0;i<m;i++) delete [] A[i];
        delete [] A;
        for(int i=0;i<k;i++) delete [] B[i];
        delete [] B;
        for(int i=0;i<m;i++) delete [] C[i];
        delete [] C;

        printf("m n k :%d %d %d\n", m, n, k);
        printf("time:%4lf s \n", end-start);
    }
    else {
        col_task = get_task(k,comm_sz,my_rank);
        int **A, ** subCol, **result;
        A = new int*[m];
        for(int i=0;i<m;i++) A[i] = new int[n];
        subCol = new int*[col_task.count];
        for(int i=0;i<col_task.count;i++) subCol[i] = new int[n];
        result = new int*[m];
        for(int i=0;i<m;i++) result[i] = new int[col_task.count];

        // 接受数据
        // 行
        for(int i=0;i<m;i++)
            MPI_Recv(A[i],n,MPI_INT,0,my_rank,MPI_COMM_WORLD,NULL);
        // 列
        for(int i=0;i<col_task.count;i++)
            MPI_Recv(subCol[i],n,MPI_INT,0,my_rank,MPI_COMM_WORLD,NULL);

        // 计算
        for(int i=0;i<m;i++){           
            for(int j=0;j<col_task.count;j++){
                int sum = 0;
                for(int l=0;l<n;l++){
                    sum += A[i][l]*subCol[j][l];
               }
               result[i][j] = sum;
            }
        }


        // 发送结果
        for(int i=0;i<m;i++)
            MPI_Send(result[i],col_task.count,MPI_INT,0,my_rank,MPI_COMM_WORLD);

        // 资源回收
        for(int i=0;i<m;i++) delete [] A[i];
        delete [] A;
        for(int i=0;i<col_task.count;i++) delete [] subCol[i];
        delete [] subCol;
        for(int i=0;i<m;i++) delete [] result[i];
        delete [] result;
    }

    MPI_Finalize();

    return 0;
}