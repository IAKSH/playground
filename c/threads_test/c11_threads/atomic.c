#include <stdio.h>
#include <stdbool.h>
#include <threads.h>
#include <stdatomic.h>

atomic_bool ready = ATOMIC_VAR_INIT(false);

int producer(void *arg) {
    //struct timespec ts = {1, 0};
    //thrd_sleep(&ts, NULL); 

    printf("Producer: Setting ready to true.\n");
    atomic_store(&ready, true); // 设置 ready 为 true
    return 0;
}

int consumer(void *arg) {
    while (!atomic_load(&ready)) { // 等待 ready 变为 true
        printf("Consumer: yield\n");
        thrd_yield(); // 主动让出 CPU
    }
    printf("Consumer: Ready detected, proceeding.\n");
    return 0;
}

int main() {
    thrd_t prod, cons;
    thrd_create(&prod, producer, NULL);
    thrd_create(&cons, consumer, NULL);

    thrd_join(prod, NULL);
    thrd_join(cons, NULL);

    return 0;
}
