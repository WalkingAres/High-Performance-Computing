void parallel_for(int start, int end, int stride, void * (*function)(void *), void * arg, int num_threads);