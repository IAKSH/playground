#include <stdio.h>
#include <threads.h>
#include <time.h>

int long_running_task(void *arg) {
    int thread_id = *(int *)arg;
    for (int i = 0; i < 3; i++) {
        printf("Thread %d is running step %d...\n", thread_id, i + 1);
        thrd_yield(); // 主动让出 CPU 控制权，给其他线程运行的机会
    }
    return 0;
}

int sleep_task(void *arg) {
    int thread_id = *(int *)arg;
    struct timespec ts = {2, 0}; // 休眠 2 秒钟
    printf("Thread %d is going to sleep...\n", thread_id);
    thrd_sleep(&ts, NULL); // 调用 thrd_sleep 休眠指定的时间
    printf("Thread %d woke up from sleep.\n", thread_id);
    return 0;
}

int main() {
    thrd_t thread1, thread2, thread3;

    int id1 = 1;
    thrd_create(&thread1, long_running_task, &id1);
    int id2 = 2;
    thrd_create(&thread2, sleep_task, &id2);
    int id3 = 3;
    thrd_create(&thread3, long_running_task, &id3);

    // 用thrd_equal比较线程句柄是否相同
    if (thrd_equal(thread1, thread3)) {
        printf("Thread 1 and Thread 3 are the same thread.\n");
    } else {
        printf("Thread 1 and Thread 3 are different threads.\n");
    }

    thrd_join(thread1, NULL);
    thrd_join(thread2, NULL);
    thrd_join(thread3, NULL);

    printf("All threads have completed.\n");
    return 0;
}
