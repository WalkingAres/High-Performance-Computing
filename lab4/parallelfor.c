#include "parallelfor.h"
#include <pthread.h>
#include <stdlib.h>

struct arg_index
{
    void * arg;
    int first;
    int last;
    int stride;
};


void parallel_for(int start, int end, int stride, void * (*function)(void *), void * arg, int num_threads) {
    // 平均分配任务
    // 总任务量
    int total = (end - start) / stride;
    int q = total /  num_threads;
    int r = total %  num_threads;
    // 每个线程的任务
    int count;

    pthread_t * threads = malloc(sizeof(pthread_t)*num_threads);
    struct arg_index * parg = malloc(sizeof(struct arg_index)*num_threads);
    for(int i=0; i<num_threads; i++) {
        // 初始化线程入口参数 （任务分配）
        if(i < r) {
            count = q + 1;
            parg[i].first = i*count;
        }
        else {
            count = q;
            parg[i].first = i*count + r;
        }
        parg[i].last = parg[i].first + count;
        parg[i].stride = stride;
        // 创建子线程
        pthread_create(threads+i,NULL,function,parg+i);
    }

    for (int i = 0; i < num_threads; i++){
        // 等待子线程运行结束
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(parg);

}