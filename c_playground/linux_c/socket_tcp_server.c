/*
参考： https://blog.csdn.net/weixin_42307601/article/details/130667724
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define MAX_CONNECTIONS 5
#define BUFFER_SIZE 1024
#define PORT 8080

int main(void) {
    // 创建服务端socket
    int server_socket_id,client_socket_id;
    if((server_socket_id = socket(AF_INET,SOCK_STREAM,0)) == 0) {
        perror("can't create socket");
	    exit(EXIT_FAILURE);
    }

    // 设置服务端监听地址
    struct sockaddr_in server_addr,client_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // 将服务端地址绑定到socket
    if(bind(server_socket_id,(struct sockaddr*)&server_addr,sizeof(server_addr)) == -1) {
	    perror("bind failed");
	    exit(EXIT_FAILURE);
    }

    // 开启对服务端socket的监听
    if(listen(server_socket_id,MAX_CONNECTIONS) == -1) {
        perror("failed to listen socket");
	    exit(EXIT_FAILURE);
    }

    printf("server listening on port %d\n",PORT);

    socklen_t client_addrlen = sizeof(client_addr);
    int bytes_received;
    char buffer[BUFFER_SIZE];
    const char* response = "nihao";

    while(1) {
        // 等待连接
        client_socket_id = accept(server_socket_id,(struct sockaddr*)&client_addr,&client_addrlen);
        if(client_socket_id == -1) {
            perror("accept error");
	        continue;
	    }

        // 接收数据
        bytes_received = recv(client_socket_id,buffer,BUFFER_SIZE,0);
	    if(bytes_received == -1) {
            perror("receive error");
	        close(client_socket_id);
	        continue;
	    }

	    printf("Received %d bytesfrom client: %s\n",bytes_received,buffer);

        // 发送数据
	    if(send(client_socket_id,response,strlen(response),0) == -1) {
            perror("send response error");
	        close(client_socket_id);
	        continue;
	    }

	    printf("send response to client: %s\n",response);

        // 关闭与客户端的连接
	    close(client_socket_id);
	    printf("client connection closed\n");
    }

    close(server_socket_id);
    return 0;
}
