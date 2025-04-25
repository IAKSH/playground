`C11`的`threads.h`不提供信号量机制(semaphore)，考虑使用`cnd_t`结合`mtx_t`做替代。

另外，如果非要使用，可以考虑OS API，比如`POSIX API`中的`<semaphore.h>`

`C11`提供了`<stdatomic.h>`用于实现原子操作。

另外，虽然`C11`引入了大量的多线程工具，但是C的大量基础IO设施依然不是线程安全的，考虑加锁或独立缓冲区，亦或是直接采用其他实现。