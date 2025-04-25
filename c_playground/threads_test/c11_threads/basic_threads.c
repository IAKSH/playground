#include <stdio.h>
#include <threads.h>
#include <time.h>

int thread_foo(void* arg) {
    int id = *(int*)arg;
    printf("thread %d is running\n",id);
    
    struct timespec ts = {1,0};
    thrd_sleep(&ts,NULL);
    thrd_yield();// 显示让出CPU

    printf("thread %d is done\n",id);
    return id;
}

#define NUM_THREADS 3

int main(void){
    thrd_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    for(int i = 0;i < NUM_THREADS;i++) {
        ids[i] = i + 1;
        if(thrd_create(&threads[i],thread_foo,&ids[i]) != thrd_success)
            fprintf(stderr,"failed to create thread %d\n",i);
    }

    for(int i = 0;i < NUM_THREADS;i++) {
        int res;
        thrd_join(threads[i],&res);
        printf("thread %d return = %d\n",i + 1,res);
    }

    return 0;
}
