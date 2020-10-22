#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <sys/time.h>

int *A;
int arrSize = 1000;
int global_index = 0;

int sum;

pthread_mutex_t mutex;

void * thread_sum(void * rank) {

    int value;
    while (global_index < arrSize) {
        pthread_mutex_lock(&mutex);     // 进临界区加锁
        sum += A[global_index];
        global_index++;
        pthread_mutex_unlock(&mutex);   // 出临界区解锁
    }

    return NULL;
}

int main(void) {
    A = new int[arrSize];
    for(int i=0; i < arrSize;i++) {
        srand(i);
        A[i] = rand() % 10;
    }

    printf("input the number of threads to work : ");
    int n;
    scanf("%d",&n);

    pthread_t *thread;
    thread = new pthread_t[n];

    struct timeval start, end;
    gettimeofday( &start, NULL );

    pthread_mutex_init(&mutex,NULL);
    for(int i=0; i<n;i++) pthread_create(thread+i,NULL,thread_sum,NULL);
    for(int i=0; i<n;i++) pthread_join(thread[i],NULL);

    gettimeofday( &end, NULL );

    double  timeuse = (1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec)/(double)1000;

    printf("sum : %d\n",sum);
    printf("time : %lf ms\n",timeuse);

    delete []thread;
    delete []A;
    pthread_mutex_destroy(&mutex);


    return 0;

}