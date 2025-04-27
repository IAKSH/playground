#pragma once

#include <driver/i2c.h>
#include <stdint.h>
#include "kalman.h"
 
bool mpu6050_init(i2c_port_t i2c_num);                       //初始化
void mpu6050_get_accel(i2c_port_t i2c_num,int16_t *accel_array);   //读取加速度计原始数据
void mpu6050_get_gryo(i2c_port_t i2c_num,int16_t *gyro_array);     //读取陀螺仪原始数据
void mpu6050_get_accel_val(i2c_port_t i2c_num,float *accel_value);  //读取转换后的加速度计数值
void mpu6050_get_gryo_val(i2c_port_t i2c_num,float *gyro_value);    //读取转换后的陀螺仪数值
void mpu6050_get_temperature(i2c_port_t i2c_num,float* temperature);

typedef struct {
    KalmanState kalman_roll;
    KalmanState kalman_pitch;
    KalmanState kalman_yaw;
    KalmanState accel_x;
    KalmanState accel_y;
    KalmanState accel_z;
    KalmanState temperature;
} Mpu6050KalmanState;

void mpu6050_kalman_init(Mpu6050KalmanState* kalman_state);
void mpu6050_kalman_update(i2c_port_t i2c_num,Mpu6050KalmanState* kalman_state, float* euler, float* accel, float* temperature);