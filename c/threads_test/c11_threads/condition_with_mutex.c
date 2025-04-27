#include <stdio.h>
#include <threads.h>

mtx_t mutex;
cnd_t cond;
int ready = 0;

int producer(void *arg) {
    struct timespec ts = {3,0};
    thrd_sleep(&ts,NULL);

    mtx_lock(&mutex);
    ready = 1;           // 更新共享状态
    printf("producer: notifying consumer\n");
    cnd_signal(&cond);   // 唤醒等待的消费者
    mtx_unlock(&mutex);
    return 0;
}

int consumer(void *arg) {
    mtx_lock(&mutex);
    while (!ready) {
        printf("consumer: wait for notification\n");
        cnd_wait(&cond, &mutex); // 等待条件改变
    }
    printf("consumer: notified, working\n");
    mtx_unlock(&mutex);
    return 0;
}

int main(void) {
    thrd_t prod, cons;

    mtx_init(&mutex, mtx_plain);
    cnd_init(&cond);

    thrd_create(&prod, producer, NULL);
    thrd_create(&cons, consumer, NULL);

    thrd_join(prod, NULL);
    thrd_join(cons, NULL);

    mtx_destroy(&mutex);
    cnd_destroy(&cond);
    return 0;
}
