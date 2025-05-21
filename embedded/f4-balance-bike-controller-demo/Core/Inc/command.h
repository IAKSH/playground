#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    COMMAND_MOVE,COMMAND_VOLT,COMMAND_CAM_ROTATE,COMMAND_CAM_SHOT,COMMAND_PID
} __CommandType;

typedef enum {
    GRAY_SCALE,R8G8B8,R5G6B5,R5G5B5,R4G4B4
} __ColorFormat;

typedef uint8_t CommandType;
typedef uint8_t ColorFormat;

typedef struct {
    uint8_t version;
    uint8_t length;
    uint8_t seq;
    CommandType type;
    // NRF24L01+自带CRC校验
    //uint16_t crc;
    union {
        struct {
            uint16_t speed[2];
        } move;
        struct {
            float angle[2];
        } cam_rotate;
        struct {
            uint16_t size[2];
            uint8_t color_format;
        } cam_shot;
        struct {
            bool write;
            bool copy_state;
            struct val {
                float kp,ki,kd;
            } angle_pid,speed_pid;
        } pid;
    } payload;
} CommandPacket;