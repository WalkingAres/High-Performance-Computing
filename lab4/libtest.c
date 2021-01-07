#include <stdio.h>
#include "parallelfor.h"

int A[12];
int B[12];
int C[12];


struct arg_index
{
    int first;
    int last;
    int stride;
};  



void * code(void * index) {
    struct arg_index *pindex = (struct arg_index *) index;
    //printf("first:%d\n",pindex->first);
    for(int i=pindex->first; i < pindex->last; i += pindex->stride){
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    for(int i=0;i<12;i++) {
        A[i] = i;
        B[i] = i;
    }


    parallel_for(0,11,1,code,NULL,4);
    printf("A:\n");
    for(int i=0;i<12;i++) printf("%d ", A[i]);
    printf("\n");
    printf("B:\n");
    for(int i=0;i<12;i++) printf("%d ", B[i]);
    printf("\n");
    printf("result:\n");
    for(int i=0;i<12;i++) printf("%d ", C[i]);
    printf("\n");

}