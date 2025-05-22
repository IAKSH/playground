#include "driver/nrf24l01p.h"
#include "utils/pid.h"
#include "wireless.h"
#include "main.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint8_t rx_address[5] = {0x0, 0x0, 0x0, 0x0, 0x01};

static void command_move_handler(CommandPacket command) {
    printf("speed: {%d, %d}\n", command.payload.move.speed[0], command.payload.move.speed[1]);
}

static void command_pid_handler(CommandPacket command) {
    if(command.payload.pid.write) {
        extern PID pid_speed,pid_angle;
        pid_speed.Kp = command.payload.pid.speed_pid.kp;
        pid_speed.Ki = command.payload.pid.speed_pid.ki;
        pid_speed.Kd = command.payload.pid.speed_pid.kd;

        pid_angle.Kp = command.payload.pid.angle_pid.kp;
        pid_angle.Ki = command.payload.pid.angle_pid.ki;
        pid_angle.Kd = command.payload.pid.angle_pid.kd;
    }
    else {
        nrf24l01p_set_mode_tx(2500, NRF24L01P_AIR_DATA_RATE_1Mbps);
        nrf24l01p_set_tx_addr(rx_address,5);

        // TODO:

        nrf24l01p_set_mode_rx(2500, NRF24L01P_AIR_DATA_RATE_1Mbps);
        //nrf24l01p_set_rx_addr(0,rx_address,5);
    }
}

void control_task(void* arg) {
    nrf24l01p_set_mode_rx(2500, NRF24L01P_AIR_DATA_RATE_1Mbps);

    if (!nrf24l01p_check()) {
        printf("nrf24l01+ no response\n");
        Error_Handler();
    }

    nrf24l01p_set_rx_addr(0,rx_address,5);

    CommandPacket command;

    while(1) {
        wireless_receive(&command,sizeof(CommandPacket));
        switch(command.type) {
        case COMMAND_MOVE:
            command_move_handler(command);
            break;
        case COMMAND_PID:
            command_pid_handler(command);
            break;
        default:
            break;
        }
    }
}
