#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

void worker(void) {
    while(1) {
        puts("pthread running");
	sleep(1);
    }
}

int main(void) {
    pthread_t id;
    int i,ret;

    ret = pthread_create(&id,NULL,(void*)worker,NULL);
    if(ret != 0) {
        puts("can't create pthread");
	exit(EXIT_FAILURE);
    }
    for(int i = 0;i < 3;i++) {
	puts("main thread running");
        sleep(1);
    }
    pthread_join(id,NULL);
    return 0;
}
