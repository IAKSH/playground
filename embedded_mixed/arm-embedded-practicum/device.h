#ifndef _DEVICE_H_
#define _DEVICE_H_

#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <asm/ioctl.h>

#define temp_9  _IO('c',0)
#define temp_10 _IO('c',1)
#define temp_11 _IO('c',2)
#define temp_12 _IO('c',3)

#define TEMP "/dev/ds18b20"

#define LED0 "/dev/led0"
#define LED1 "/dev/led1"
#define LED2 "/dev/led2"
#define LED3 "/dev/led3"
#define LEDON _IO('L',0)            //����
#define LEDOFF _IO('L',2)           //����

#define BUZZER "/dev/pwm0"
#define PWMON  _IO('P', 0)          //����������
#define PWMOFF _IO('P', 1)          //����������
#define PWMSET _IO('P', 2)

int fd;
short temp;
char zheng,fen;
float temputer,resolution;

void ds18b20_init()
{
    fd = open(TEMP,O_RDWR);
    if(fd < 0){
        perror("open");
    }
    if(ioctl(fd,temp_12,&resolution))
    {
        perror("ioctl \n");
    }
}

float get_temp()
{
    if(!read(fd,&temp,sizeof(short))){
            perror("read");
            return 0;
    }
    zheng = temp>>4;
    fen = temp & 0xf;
    if(zheng & (1<<8)){
        temputer = (temp - 65535) * resolution;
    }else{
        temputer = zheng + fen * resolution;
    }
    if ((temputer >= (-55)) && (temputer <= 125))
        return temputer;
    else
        return 0;
}

#endif

