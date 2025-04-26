#pragma once

#define GYRO_ERROR_MPU_INIT      -1
#define GYRO_ERROR_SET_SENSOR    -2
#define GYRO_ERROR_CONFIG_FIFO   -3
#define GYRO_ERROR_SET_RATE      -4
#define GYRO_ERROR_LOAD_MOTION_DRIVER    -5 
#define GYRO_ERROR_SET_ORIENTATION       -6
#define GYRO_ERROR_ENABLE_FEATURE        -7
#define GYRO_ERROR_SET_FIFO_RATE         -8
#define GYRO_ERROR_SELF_TEST             -9
#define GYRO_ERROR_DMP_STATE             -10

#define DEFAULT_MPU_HZ  200
#define Q30  1073741824.0f

int gyro_init(void);
int gyro_get_data(float *pitch, float *roll, float *yaw);
