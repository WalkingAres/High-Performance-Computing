#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <cmath>
#include <semaphore.h>

struct thread_arg
{
    double a;
    double b;
    double c;
    double * result;
};

struct result
{
    double *a;
    double *b;
    double *x1;
    double *x2;
};


typedef thread_arg targ;
typedef result result;

pthread_t t1, t2, t3;
pthread_mutex_t mutex1, mutex2;
pthread_cond_t cond1, cond2;
int r1_flag =0 , r2_flag = 0;

void * thread1(void * argv) {
    pthread_mutex_lock(&mutex1);
    targ * t = (targ *) argv;
    *(t->result) = - t->b / (2 * t->a);
    r1_flag++;
    pthread_cond_signal(&cond1);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void * thread2(void * argv) {
    pthread_mutex_lock(&mutex2);
    targ * t = (targ *) argv;
    *(t->result) = sqrt(t->b * t->b - 4*t->a*t->c) / (2*t->a);
    r2_flag ++ ;
    pthread_cond_signal(&cond2);
    pthread_mutex_unlock(&mutex2);
    return NULL;
}

void * thread3(void * argv) {

    result * r = (result *) argv;
    pthread_mutex_lock(&mutex1);
    if(r1_flag <= 0)
    pthread_cond_wait(&cond1,&mutex1);
    double a = *(r->a);
    pthread_mutex_unlock(&mutex1);

    pthread_mutex_lock(&mutex2);
    if(r2_flag <= 0) pthread_cond_wait(&cond2,&mutex2);
    double b = *(r->b);
    pthread_mutex_unlock(&mutex2);

    *(r->x1) = a + b;
    *(r->x2) = a - b;

    return NULL;
}

int main(void) {
    std::cout << "input a , b , c: ";
    double a, b, c;
    std::cin >> a >> b >> c;

    double r1, r2, x1, x2;
    targ targ1 = {a,b,c,&r1}, targ2 = {a,b,c,&r2};
    result r = {&r1,&r2,&x1,&x2};

    pthread_cond_init(&cond1, NULL);
    pthread_cond_init(&cond2, NULL);
    pthread_mutex_init(&mutex1, NULL);
    pthread_mutex_init(&mutex2, NULL);


    pthread_create(&t1,NULL,thread1,&targ1);
    pthread_create(&t2,NULL,thread2,&targ2);
    pthread_create(&t3,NULL,thread3,&r);

    pthread_join(t1,NULL);
    pthread_join(t2,NULL);
    pthread_join(t3,NULL);

    std::cout << "the solution is :\n" << "x1 : " << x1 << "\nx2 : " << x2 <<"\n";
    return 0;

}