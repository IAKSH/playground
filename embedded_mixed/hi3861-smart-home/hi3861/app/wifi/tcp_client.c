#include "tcp_client.h"

#include <stdio.h>
#include <unistd.h>
#include "cmsis_os2.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

extern osMutexId_t dht11_mutex_id;
extern unsigned int dht11_data[4];

#define TCP_SERVER_PORT 8080
#define TCP_SERVER_ADDR "192.168.10.8"

void tcp_client_startup(void) {
    int sock = socket(AF_INET,SOCK_STREAM,0);
    if(sock < 0) {
        printf("[tcp_client] failed to create a socket!\n");
        return;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr,0,sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(TCP_SERVER_PORT);
    server_addr.sin_addr.s_addr = inet_addr(TCP_SERVER_ADDR);

    
    while(connect(sock,(struct sockaddr*)&server_addr,sizeof(server_addr)) < 0) {
        printf("[tcp_client] connection failed, retry in 1s...\n");
        osDelay(100);
        //close(sock);
        //return;
    }

    char *message = "Hello TCP Server, I'm Hi3861!";
    send(sock,message,strlen(message),0);

    char s[48];
    int _dht11_data[4];

    while(1) {
        osMutexAcquire(dht11_mutex_id, osWaitForever);
        memcpy(_dht11_data,dht11_data,sizeof(int) * 4);
        osMutexRelease(dht11_mutex_id);

        sprintf(s,"temp: %d.%d, humi: %d.%d\n", _dht11_data[2], _dht11_data[3], _dht11_data[0], _dht11_data[1]);
        send(sock,s,strlen(s),0);

        osDelay(100);
    }

    //char buffer[1024] = {0};
    //recv(sock,buffer,sizeof(buffer) - 1,0);
    //printf("[tcp_client] Received: %s\n",buffer);

    close(sock);
}