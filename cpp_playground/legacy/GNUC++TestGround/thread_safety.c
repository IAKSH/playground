#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int increament_count()
{
    static int count = 0;
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&mutex);
    ++count;
    int ret = count;
    pthread_mutex_unlock(&mutex);

    return ret
};

int main(void)
{
    pthread_t threads[9];
    for(int i = 0;i < 9;i++)
        pthread_create(threads + i,NULL,(void*)increament_count,NULL);

    for(int i = 0;i < 9;i++)
    {
        int ret;
        pthread_join(threads[i],(void*)&ret);
        printf("sub thread returned: %d\n",ret);
    }

    printf("count: %d\n",increament_count());
    return 0;
}