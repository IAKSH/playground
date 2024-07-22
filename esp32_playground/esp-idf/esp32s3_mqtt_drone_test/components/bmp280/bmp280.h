#pragma once

#include <stdint.h>
#include "driver/i2c.h"
#include "kalman.h"

bool bmp280_init(i2c_port_t i2c_num);                       // 初始化

void bmp280_get_raw_temp_press(i2c_port_t i2c_num, int32_t *temp, int32_t *press);  // 读取原始的温度和气压数据
void bmp280_get_temp_press(i2c_port_t i2c_num, int32_t *temp, uint32_t *press);          // 读取转换后的温度和气压数据
void bmp280_get_altitude(i2c_port_t i2c_num, float *altitude);
void bmp280_press_temp_to_altitude(uint32_t press,int32_t temperature,float* altitude);

typedef struct {
    KalmanState temperature,press,altitude;
} bmp280_kalman_state_t;

void bmp280_kalman_init(bmp280_kalman_state_t* kalman_state);
void bmp280_kalman_update(i2c_port_t i2c_num,bmp280_kalman_state_t* kalman_state,float* press,float* altitude,float* temperature);