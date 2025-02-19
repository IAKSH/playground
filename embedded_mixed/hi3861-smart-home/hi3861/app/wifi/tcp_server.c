#include "tcp_server.h"

#include <stdio.h>
#include <unistd.h>
#include "cmsis_os2.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

char device_name[16] = "hi3861";
char device_type[16] = "general";

#define BROADCAST_INTERVAL 5
#define UDP_PORT 12345
#define TCP_SERVER_PORT 8080

static struct sockaddr_in udp_broadcast_addr;
static int udp_sock;

static void udp_broadcast() {
    printf("[udp_broadcast] begin broadcast\n");

    char message[128];
    snprintf(message,sizeof(message),"[home_iot] name: %s, type: %s, port: %d",device_name,device_type,TCP_SERVER_PORT);

    while(1) {
        sendto(udp_sock,message,strlen(message),0,(struct sockaddr*)&udp_broadcast_addr,sizeof(udp_broadcast_addr));
        sleep(BROADCAST_INTERVAL);
    }
}

static osThreadId_t setup_udp_broadcast(void) {
    // setup udp broadcast
    udp_sock = socket(AF_INET,SOCK_DGRAM,0);
    if(udp_sock < 0) {
        perror("[udp_broadcast] UDP socket creation failed!");
        exit(EXIT_FAILURE);
    }

    int broadcast_permission = 1;
    if(setsockopt(udp_sock,SOL_SOCKET,SO_BROADCAST,&broadcast_permission,sizeof(broadcast_permission)) < 0) {
        perror("[udp_broadcast] setting broadcast option failed");
        close(udp_sock);
        exit(EXIT_FAILURE);
    }

    memset(&udp_broadcast_addr,0,sizeof(udp_broadcast_addr));
    udp_broadcast_addr.sin_family = AF_INET;
    udp_broadcast_addr.sin_port = htons(UDP_PORT);
    udp_broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);

    // begin udp broadcast
    osThreadAttr_t attr;
    attr.name = "udp_broadcast";
    attr.attr_bits = 0U;
    attr.cb_mem = NULL;
    attr.cb_size = 0U;
    attr.stack_mem = NULL;
    attr.stack_size = 4096;
    attr.priority = osPriorityBelowNormal;

    osThreadId_t thread = osThreadNew(udp_broadcast, NULL, &attr);
    if (thread == NULL) {
        printf("[udp_broadcast] Failed to create task!\n");
        close(udp_sock);
        // better not shutting down here, as we can still add this device directly using ip
        // it's just an auto-discovery, not a big deal
    }

    return thread;
}

extern osMutexId_t dht11_mutex_id;
extern unsigned int dht11_data[4];

#define BACKLOG 5

void tcp_server_startup(void) {
    osThreadId_t broadcast_thread = setup_udp_broadcast();

    // setup tcp server
    int server_sock = socket(AF_INET,SOCK_STREAM,0);
    if(server_sock < 0) {
        printf("[tcp_server] failed to create a socket!\n");
        return;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr,0,sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(TCP_SERVER_PORT);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if(bind(server_sock,(struct sockaddr*)&server_addr,sizeof(server_addr)) < 0) {
        printf("[tcp_server] bind failed!\n");
        close(server_sock);
        return;
    }

    if(listen(server_sock,BACKLOG) < 0) {
        printf("[tcp_server] listen failed!\n");
        close(server_sock);
        return;
    }

    printf("[tcp_server] listening on port %d\n",TCP_SERVER_PORT);

    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    int client_sock;

    osStatus_t status;

    while((client_sock = accept(server_sock,(struct sockaddr*)&client_addr,&client_addr_len)) >= 0) {
        printf("[tcp_server] client connected\n");

        printf("[udp_broadcast] suspended for incoming tcp connection\n");
        status = osThreadSuspend(broadcast_thread);
        if(status != osOK) {
            perror("[udp_broadcast] failed to suspend udp_broadcast thread!");
            exit(EXIT_FAILURE);
        }

        int _dht11_data[4];
        char str_buffer[48];

        while(1) {
            osMutexAcquire(dht11_mutex_id, osWaitForever);
            memcpy(_dht11_data,dht11_data,sizeof(int) * 4);
            osMutexRelease(dht11_mutex_id);

            sprintf(str_buffer,"temp: %d.%d, humi: %d.%d\n", _dht11_data[2], _dht11_data[3], _dht11_data[0], _dht11_data[1]);
            if (send(client_sock, str_buffer, strlen(str_buffer), 0) < 0) {
                printf("[tcp_server] client disconnected\n");
                close(client_sock);
                break;
            }

            osDelay(100);
        }

        printf("[udp_broadcast] resumed for incoming tcp connection\n");
        status = osThreadResume(broadcast_thread);
        if(status != osOK) {
            perror("[udp_broadcast] failed to resume udp_broadcast thread!");
            exit(EXIT_FAILURE);
        }
    }
    
    printf("[tcp_server] server shutdown\n");
    close(server_sock);
}