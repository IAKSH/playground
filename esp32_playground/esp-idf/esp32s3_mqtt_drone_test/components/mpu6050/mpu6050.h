#pragma once

#include <stdint.h>
#include "driver/i2c.h"
 
bool mpu6050_init(i2c_port_t i2c_num);
void mpu6050_get_accel(i2c_port_t i2c_num,int16_t *accel_array);
void mpu6050_get_gyro(i2c_port_t i2c_num,int16_t *gyro_array); 
void mpu6050_get_accel_val(i2c_port_t i2c_num,float *accel_value);
void mpu6050_get_gyro_val(i2c_port_t i2c_num,float *gyro_value);
void mpu6050_get_temperature(i2c_port_t i2c_num,float* temperature);

typedef struct {
    float angle;       // 角度
    float bias;        // 偏差
    float rate;        // 角速度
    float P[2][2];     // 协方差矩阵
} mpu6050_kalman_state_t;

void mpu6050_kalman_init(mpu6050_kalman_state_t* state, float angle, float bias, float rate);
void mpu6050_kalman_update(mpu6050_kalman_state_t* state, float gyro_rate, float accel_angle, float dt);