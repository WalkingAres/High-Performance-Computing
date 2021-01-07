#ifndef PARALLELFOR_H
#define PARALLELFOR_H

struct thread_arg
{
    void *arg;
    int first;
    int last;
    int stride;
};

typedef struct thread_arg targ;


void parallel_for(int start, int end, int stride, void * (*function)(void *), void * arg, int num_threads);

#endif