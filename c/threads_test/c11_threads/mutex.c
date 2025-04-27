#include <stdio.h>
#include <threads.h>

#define NUM_THREADS 5
#define INCREMENTS 10000

mtx_t mutex;
int counter = 0;

int increment(void* arg) {
    for(int i = 0;i < INCREMENTS;i++) {
        mtx_lock(&mutex);
        counter++;
        mtx_unlock(&mutex);
    }
    return 0;
}

int main(void) {
    thrd_t threads[NUM_THREADS];

    if(mtx_init(&mutex,mtx_plain) != thrd_success) {
        fprintf(stderr,"failed to init mutex\n");
        return 1;
    }

    for(int i = 0;i < NUM_THREADS;i++)
        thrd_create(&threads[i],increment,NULL);

    for(int i = 0;i < NUM_THREADS;i++) {
        thrd_join(threads[i],NULL);
    }

    printf("counter = %d\n",counter);
    mtx_destroy(&mutex);
    return 0;
}