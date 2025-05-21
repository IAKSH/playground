#include "driver/nrf24l01p.h"
#include "command.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint8_t rx_address[5] = {0x0,0x0,0x0,0x0,0x01};

static const char* command_type_to_str(CommandType type) {
    switch(type) {
    case COMMAND_MOVE:
        return "move";
    case COMMAND_VOLT:
        return "volt";
    case COMMAND_CAM_ROTATE:
        return "cam_rotate";
    case COMMAND_CAM_SHOT:
        return "cam_shot";
    case COMMAND_PID:
        return "pid";
    default:
        return "unkonwn";
    }
}

static void command_move_handler(CommandPacket command) {
    printf("speed: {%d,%d}\n",command.payload.move.speed[0],command.payload.move.speed[1]);
}

void control_task(void* arg) {
    nrf24l01p_rx_init(2500,_1Mbps);

    if(!nrf24l01p_check()) {
        printf("nrf24l01+ no response\n");
        Error_Handler();
    }

    nrf24l01p_set_rx_addr(0,rx_address,5);

    CommandPacket command;
    while(1) {
        osMessageQueueGet(command_queue,&command,NULL,osWaitForever);
        printf("command recieved: type = %s\n",command_type_to_str(command.type));
        switch(command.type) {
        case COMMAND_MOVE:
            command_move_handler(command);
            break;
        default:
            break;
        }
    }
}