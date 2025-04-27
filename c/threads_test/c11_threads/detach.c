#include <stdio.h>
#include <threads.h>

int detached_thread(void *arg) {
    for (int i = 0; i < 3; i++) {
        printf("detached thread has been executed for %g sec\n", (i + 1) / 2.0f);
        struct timespec ts = {0, 500 * 1000 * 1000}; // 0.5 秒
        thrd_sleep(&ts, NULL);
    }
    printf("all threads detached\n");
    return 0;
}

int main() {
    thrd_t thr;
    thrd_create(&thr, detached_thread, NULL);
    
    // 将线程设置为分离状态
    thrd_detach(thr);

    // 主线程可以在此继续执行其他任务，不必等待分离线程结束
    printf("main thread running...\n");
    struct timespec ts_main = {2, 0};
    thrd_sleep(&ts_main, NULL);  // 主线程等待 2 秒以确保分离线程能执行完毕

    return 0;
}
