#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int hit_sum = 0;
pthread_mutex_t mutex;

void * thread(void * times) {
    int total = *(int *) times;
    double x, y;
    int a, b;
    int hit = 0;
    for(int i=0;i<total;i++) {
        a = rand();
        b = rand();
        x = (a%10001) / 10001.0;
        y = (b%10001) / 10001.0;

        if( y <= x*x) hit++;
    }

    pthread_mutex_lock(&mutex);
    hit_sum += hit;
    pthread_mutex_unlock(&mutex);

    return NULL;
}


int main(int argc , char *argv[]) {

    if( argc < 2) {
        printf("input the total times and number of threads , try again!\n");
        return 0;
    }

    int total = atoi(argv[1]);
    int num_thread = atoi(argv[2]);
    int times = total / num_thread;

    pthread_mutex_init(&mutex,NULL);
    pthread_t *threads = malloc(num_thread*sizeof(pthread_t));
    for(int i=0;i<num_thread;i++) {
        pthread_create(threads+i,NULL,thread,&times);
    }

    for(int i=0;i<num_thread;i++) {
        pthread_join(threads[i],NULL);
    }

    double result = 1.0*hit_sum/(times*num_thread);
    printf("result:%lf\n",result);

    free(threads);
    
    return 0;
}