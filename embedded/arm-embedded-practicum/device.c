#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "device.h"

#define MAX_CONNECTIONS 5
#define BUFFER_SIZE 1024
#define PORT 8888

int main(void) {
    int server_socketfd,client_socketfd;
    if((server_socketfd = socket(AF_INET,SOCK_STREAM,0)) == 0) {
		perror("can't create socket");
		exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_addr,client_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if(bind(server_socketfd,(struct sockaddr*)&server_addr,sizeof(server_addr)) == -1) {
		perror("bind failed");
		exit(EXIT_FAILURE);
    }

    if(listen(server_socketfd,MAX_CONNECTIONS) == -1) {
        perror("failed to listen socket");
		exit(EXIT_FAILURE);
    }

    printf("server listening on port %d\n",PORT);

    socklen_t client_addr_len = sizeof(client_addr);
    
    while(1) {
        client_socketfd = accept(server_socketfd,(struct sockaddr*)&client_addr,&client_addr_len);
		if(client_socketfd == -1) {
	    	perror("accept error");
	    	continue;
		}

		int pid = fork();
		if(pid == 0) {
	    	ds18b20_init();
	    	float temp;
	    	int buzzerfd = open(BUZZER,O_RDWR);
	    	char buffer[BUFFER_SIZE];
	    	while(1) {
				memset(buffer,BUFFER_SIZE,'\0');
				temp = get_temp();
				printf("read in temp: %.2f\n",temp);
				sprintf(buffer,"%.2f",temp);
	        	if(send(client_socketfd,&buffer,BUFFER_SIZE,0) == -1) {
	            	perror("send data error");
	            	close(client_socketfd);
	            	break;
				}
				ioctl(buzzerfd,temp >= 40.0f ? PWMON : PWMOFF);
				sleep(1);
	    	}
		}
		else {
	    	int led0fd = open(LED0,O_RDWR);
	    	int led1fd = open(LED1,O_RDWR);
	    	int led2fd = open(LED2,O_RDWR);
	    	int led3fd = open(LED3,O_RDWR);
	    	int bytes_received;
	    	char buffer[BUFFER_SIZE];
	    	while(1) {
				memset(buffer,BUFFER_SIZE,'\0');
	        	bytes_received = recv(client_socketfd,buffer,BUFFER_SIZE,0);
	        	if(bytes_received == -1) {
                    perror("receive error");
		    		close(client_socketfd);
		    		break;
	        	}
	        	printf("received data: %s\n",buffer);

				int on = (strcmp(buffer + 5,"onf") == 0);
				printf("on = %d\n",on);
				printf("buffer[3] = %c\n",buffer[3]);
				switch(buffer[3]) {
				case '0':
		    		ioctl(led0fd,on ? LEDON : LEDOFF);
		    		break;
				case '1':
		    		ioctl(led1fd,on ? LEDON : LEDOFF);
		    		break;
				case '2':
		    		ioctl(led2fd,on ? LEDON : LEDOFF);
		    		break;
				case '3':
		    		ioctl(led3fd,on ? LEDON : LEDOFF);
		    		break;
				}
	    	}
		}
    }
}
