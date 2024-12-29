#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//#include <sys/wait.h>

int main() {
    puts("hello world!");

    int pipefd[2];
    if(pipe(pipefd)) {
        perror("failed to open pipe");
	    exit(EXIT_FAILURE);
    }

    int pid = fork();

    printf("fork returns %d\n",pid);
    if(pid == -1) {
        perror("can't create sub process");
	    exit(EXIT_FAILURE);
    }
    else if(pid == 0) { 
	    printf("running sub-process pid:%d ppid:%d\n",getpid(),getppid());
        int a,b;
	    scanf("%d%d",&a,&b);
	    write(pipefd[1],&a,sizeof(int));
	    write(pipefd[1],&b,sizeof(int));
	    printf("sub-process pid:%d ppid:%d exited\n",getpid(),getppid());
    }
    else {
	    //wait(NULL);
	    printf("running main process pid:%d\n",getpid());
	    int a,b;
	    read(pipefd[0],&a,sizeof(int));
	    read(pipefd[0],&b,sizeof(int));
	    printf("a + b = %d + %d = %d\n",a,b,a + b);
	    close(pipefd[0]);
	    close(pipefd[1]);
	    puts("main process exiting");
    }

    return 0;
}
