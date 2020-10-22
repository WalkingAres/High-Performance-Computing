#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <sys/time.h>

int *A;
int arrSize = 1000;
int global_index = 0;

int sum;

pthread_mutex_t mutex_index;
pthread_mutex_t mutex_sum;

void * thread_sum(void * rank) {

    int size = 10000;          // 每次的计算任务
    while (global_index < arrSize) {
        pthread_mutex_lock(&mutex_index);
        if(global_index + size >= arrSize) size = arrSize-global_index;
        global_index += size;
        pthread_mutex_unlock(&mutex_index);

        int mysum = 0;
        for(int i=0;i<size;i++) mysum += A[global_index-size+i]; 

        pthread_mutex_lock(&mutex_sum);
        sum += mysum;
        pthread_mutex_unlock(&mutex_sum);
    }

    return NULL;
}

int main(void) {

    printf("input the array size and the number of threads to work : ");
    int n;
    scanf("%d %d",&arrSize, &n);
    
    A = new int[arrSize];
    for(int i=0; i < arrSize;i++) {
        srand(i);
        A[i] = rand() % 10;
    }

    pthread_mutex_init(&mutex_index,NULL);
    pthread_mutex_init(&mutex_sum,NULL);
    
    pthread_t *thread;
    thread = new pthread_t[n];

    struct timeval start, end;
    gettimeofday( &start, NULL );

    for(int i=0; i<n;i++) pthread_create(thread+i,NULL,thread_sum,NULL);
    for(int i=0; i<n;i++) pthread_join(thread[i],NULL);

    gettimeofday( &end, NULL );

    double  timeuse = (1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec)/(double)1000;

    printf("sum : %d\n",sum);
    printf("time : %lf ms\n",timeuse);

    FILE *a;
    a = fopen("vectorA","wb");
    for(int i=0;i<arrSize;i++ ) fprintf(a,"%d ", A[i]);
    fclose(a);

    pthread_mutex_destroy(&mutex_index);
    pthread_mutex_destroy(&mutex_sum);



    return 0;

}