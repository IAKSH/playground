#include "tasks.h"
#include "main.h"
#include "status.hpp"
#include "battery.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <array>
#include <algorithm>

static void input(char* buf,uint16_t len) {
    uint16_t i = 0;
    for(;i < len;i++) {
        HAL_StatusTypeDef  res = HAL_UART_Receive(&huart1,(uint8_t*)buf + i,1,10);
        if(res == HAL_TIMEOUT) {
            osDelay(1);
            i--;
            continue;
        }
        else if (res != HAL_OK)
            Error_Handler();
        if(buf[i] == '\n' || buf[i] == '\r' || buf[i] == '\0')
            break;
    }
    memset(buf + i,'\0',len - i);
}

//static void output(char* buf,uint16_t len) {
//    uint16_t i = 0;
//    for(;i < len;i++) {
//        if(buf[i] == '\0')
//            break;
//    }
//    HAL_UART_Transmit(&huart1,(uint8_t*)buf,i,HAL_MAX_DELAY);
//}

static const std::array<std::pair<const char*,void(*)()>,15> COMMANDS {
    std::pair{"help",[](){
        printf("no help for you!\n");
    }},
    std::pair{"led_off",[](){
        //set_led_status(LedStatus::OFF);
        LedStatus status = LedStatus::OFF;
        osMessageQueuePut(led_message_queue,&status,0,0);
    }},
    std::pair{"led_on",[](){
        //set_led_status(LedStatus::KEEP);
        LedStatus status = LedStatus::KEEP;
        osMessageQueuePut(led_message_queue,&status,0,0);
    }},
    std::pair{"led_rgb",[](){
        //set_led_status(LedStatus::RGB);
        LedStatus status = LedStatus::RGB;
        osMessageQueuePut(led_message_queue,&status,0,0);
    }},
    std::pair{"led_beep",[](){
        //set_led_status(LedStatus::BEEP);
        LedStatus status = LedStatus::BEEP;
        osMessageQueuePut(led_message_queue,&status,0,0);
    }},
    std::pair{"volt",[](){
        float volt = 0.0f;
        constexpr uint8_t CNT{ 10 };
        for(int i = 0;i < CNT;i++)
            volt += car::get_battery_volt();
        volt /= CNT;
        printf("%d.%d%dv\n",
            static_cast<int>(volt),
            static_cast<int>(volt * 10) % 10,
            static_cast<int>(volt * 100) % 10
        );
    }},
    std::pair{"stop",[](){
        DriveStatus status = DriveStatus::STOP;
        osMessageQueuePut(drive_message_queue,&status,0,0);
    }},
    std::pair{"auto",[](){
        DriveStatus status = DriveStatus::AUTO;
        osMessageQueuePut(drive_message_queue,&status,0,0);
    }},
    std::pair{"ir_test",[](){
        DriveStatus status = DriveStatus::IR_TEST;
        osMessageQueuePut(drive_message_queue,&status,0,0);
    }},
    std::pair{"manual",[](){
        DriveStatus status = DriveStatus::MANUAL;
        osMessageQueuePut(drive_message_queue,&status,0,0);
    }},
    std::pair{"drive_forward",[](){
        ManualDriveCommand command = ManualDriveCommand::FORWARD;
        osMessageQueuePut(manual_drive_message_queue,&command,0,0);
    }},
    std::pair{"drive_backward",[](){
        ManualDriveCommand command = ManualDriveCommand::BACKWORD;
        osMessageQueuePut(manual_drive_message_queue,&command,0,0);
    }},
    std::pair{"drive_left",[](){
        ManualDriveCommand command = ManualDriveCommand::TURN_LEFT;
        osMessageQueuePut(manual_drive_message_queue,&command,0,0);
    }},
    std::pair{"drive_right",[](){
        ManualDriveCommand command = ManualDriveCommand::TURN_RIGHT;
        osMessageQueuePut(manual_drive_message_queue,&command,0,0);
    }},
    std::pair{"drive_stop",[](){
        ManualDriveCommand command = ManualDriveCommand::STOP;
        osMessageQueuePut(manual_drive_message_queue,&command,0,0);
    }},
};

void control_task(void* args) {
    printf("[control] entrying\n");

    char buf[32] = {'\0'};
    bool command_found = false;
    while(true) {
        input(buf,sizeof(buf));
        //memset(buf,'\0',sizeof(buf));
        //scanf("%s",buf);
        printf("[debug] buf: %s\n",buf);
        //output(buf,sizeof(buf));
        command_found = false;
        for(const auto p : COMMANDS) {
            if(strcmp(p.first,buf) == 0) {
                (p.second)();
                command_found = true;
            }
        }
        if(!command_found)
            printf("unkown command, try help\n");
    }
}