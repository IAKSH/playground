/*
参考： https://blog.csdn.net/weixin_42307601/article/details/130667724
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main(void) {
    // 创建socket
    int socket_id = socket(AF_INET,SOCK_STREAM,0);// IPv4,TCP,Auto Protocol    
    if(socket_id < 0) {
        fprintf(stderr,"can't create socket\n");
        exit(EXIT_FAILURE);	
    }
    else
	    fprintf(stdout,"created socket, id = %d\n",socket_id);

    // 创建地址（ipv4地址及端口）
    struct sockaddr_in addr;
    memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);

    if(inet_pton(AF_INET,"127.0.0.1",&addr.sin_addr) <= 0) {
        fprintf(stderr,"invalid address\n");
	    exit(EXIT_FAILURE);
    }

    // 令socket连接到地址
    if(connect(socket_id,(struct sockaddr*)&addr,sizeof(addr)) < 0) {
        fprintf(stderr,"connection failed\n");
	    exit(EXIT_FAILURE);
    }

    // sned data
    send(socket_id,"nihaoma?",sizeof(char) * 9,0);
    // receive data
    char buffer[1024] = {0};
    int read_val = read(socket_id,buffer,sizeof(buffer));
    fprintf(stdout,"response: %s\n",buffer);

    close(socket_id);

    return 0;
}
