#include <cmsis_os2.h>
#include <cstdio>
#include <array>
#include "main.h"
#include "tasks.h"
#include "h_bridge.hpp"
#include "ir_switch.hpp"
#include "status.hpp"

static std::array<car::IRSwitch,5> ir_switches {
    car::IRSwitch(IR_SWITCH_1_GPIO_Port,IR_SWITCH_1_Pin),
    car::IRSwitch(IR_SWITCH_2_GPIO_Port,IR_SWITCH_2_Pin),
    car::IRSwitch(IR_SWITCH_3_GPIO_Port,IR_SWITCH_3_Pin),
    car::IRSwitch(IR_SWITCH_4_GPIO_Port,IR_SWITCH_4_Pin),
    car::IRSwitch(IR_SWITCH_5_GPIO_Port,IR_SWITCH_5_Pin)
};

static car::HBridgeConfig hbridge_conf {
    .tim = htim2,
    .channel_a = TIM_CHANNEL_1,
    .channel_b = TIM_CHANNEL_4,
    .control_port_a1 = MOTOR_INPUT1_GPIO_Port,
    .control_port_a2 = MOTOR_INPUT2_GPIO_Port,
    .control_port_b1 = MOTOR_INPUT3_GPIO_Port,
    .control_port_b2 = MOTOR_INPUT4_GPIO_Port,
    .control_pin_a1 = MOTOR_INPUT1_Pin,
    .control_pin_a2 = MOTOR_INPUT2_Pin,
    .control_pin_b1 = MOTOR_INPUT3_Pin,
    .control_pin_b2 = MOTOR_INPUT4_Pin
};

static car::HBridge* hbridge;

static void ir_test_loop() {
    printf("[drive] switch to ir_test\n");
    while(true) {
        for(uint8_t i = 0;i < ir_switches.size();i++) {
            if(i != 1 && ir_switches[i].activated())
                printf("[drive] ir %u activated\n",i);
        }
        if(osMessageQueueGetCount(drive_message_queue) != 0)
            break;
        osDelay(10);
    }
}

static void legacy_auto_drive_loop() {
    printf("[drive] switch to legacy_auto_drive\n");
    hbridge->speed = {350,350};
    while(true) {
        // stop if collision
        // maybe we need mutex here or maybe not
        printf("[debug] distance: %d\n",static_cast<int>(distance));
        if(distance < 40.0f) {
            printf("[drive] collision\n");
            hbridge->speed = {0,0};
        }
        else {
            // base speed
            hbridge->speed[0] = 250;
            hbridge->speed[1] = 250;
            
            // move forward and eliminate offsets
            if(ir_switches[0].activated()) {
                hbridge->speed = {-250,500};
            }
            // ir_switch 1 has been broken
            //if(ir_switches[1].activated())
            //    hbridge.speed[0] += MOTOR_SPEED_OFFSET;
            // disable for balance
            //if(ir_switches[3].activated())
            //    hbridge.speed[1] += MOTOR_SPEED_OFFSET;
            if(ir_switches[4].activated()) {
                hbridge->speed = {500,-250};
            }
        }

        hbridge->apply_speed();
        if(osMessageQueueGetCount(drive_message_queue) != 0)
            break;
        osDelay(10);
    }
}

static void manual_drive_loop() {
    printf("[drive] switch to manual_drive\n");
    ManualDriveCommand command;
    constexpr uint16_t SPEED{ 500 };
    while(true) {
        osMessageQueueGet(manual_drive_message_queue,&command,NULL,osWaitForever);
        switch(command) {
        case ManualDriveCommand::FORWARD:
            printf("[manual_drive] forward\n");
            hbridge->speed = {SPEED,SPEED};
            break;
        case ManualDriveCommand::BACKWORD:
            printf("[manual_drive] backward\n");
            hbridge->speed = {-SPEED,-SPEED};
            break;
        case ManualDriveCommand::TURN_LEFT:
            printf("[manual_drive] turn left\n");
            hbridge->speed = {-SPEED,SPEED};
            break;
        case ManualDriveCommand::TURN_RIGHT:
            printf("[manual_drive] turn right\n");
            hbridge->speed = {SPEED,-SPEED};
            break;
        case ManualDriveCommand::STOP:
            printf("[manual_drive] stop\n");
            hbridge->speed = {0,0};
            break;
        }
        hbridge->apply_speed();
        if(osMessageQueueGetCount(drive_message_queue) != 0)
            break;
        osDelay(10);
    }
}

static void motor_stop() {
    printf("[drive] stop\n");
    hbridge->speed = {0,0};
    hbridge->apply_speed();
}

void drive_task(void* args) {
    printf("[drive] entrying\n");

    car::HBridge __hbridge(hbridge_conf);
    hbridge = &__hbridge;

    DriveStatus status = DriveStatus::AUTO;
    osMessageQueuePut(drive_message_queue,&status,0,0);
    while(true) {
        osMessageQueueGet(drive_message_queue,&status,NULL,osWaitForever);
        switch(status) {
        case DriveStatus::AUTO:
            legacy_auto_drive_loop();
            break;
        case DriveStatus::MANUAL:
            manual_drive_loop();
            break;
        case DriveStatus::IR_TEST:
            ir_test_loop();
            break;
        case DriveStatus::STOP:
            motor_stop();
            break;
        }
    }
}