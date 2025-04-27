#include <stdio.h>
#include <threads.h>

tss_t key;

void destructor(void *val) {
    printf("destruct:\t%s\n", (char *)val);
}

int foo(void *arg) {
    // 为当前线程设置局部存储数据
    tss_set(key, arg);
    printf("storing:\t%s\n", (char *)tss_get(key));
    return 0;
}

int main() {
    thrd_t threads[2];

    // 创建一个 TSS 键，并指定析构函数
    if(tss_create(&key, destructor) != thrd_success) {
        fprintf(stderr, "failed to create a TSS key\n");
        return 1;
    }

    thrd_create(threads, foo, "data of thread 1");
    thrd_create(threads + 1, foo, "data of thread 2");

    for (int i = 0; i < 2; i++) {
        thrd_join(threads[i], NULL);
    }

    tss_delete(key);
    return 0;
}
