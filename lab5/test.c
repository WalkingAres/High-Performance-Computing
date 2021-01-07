#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>



int main(void) {

    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int a[6] = {-1,-1,-1,-1,-1,-1};
    a[my_rank] = my_rank;
    MPI_Allgather(a+my_rank, 1, MPI_INT, a+my_rank, 1,MPI_INT,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == 0) {
        for(int i=0;i<6;i++) printf("rank %d %d\n",my_rank,a[i]);
    }

    MPI_Finalize();
    return 0;
}