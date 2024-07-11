#pragma once

/* Attributes State Machine */
enum
{
    // Service Declaration
    IDX_SVC,
    // MPU6050
    /* Characteristic Declaration */
    MPU6050_CHAR_ACCEL_X,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_ACCEL_X,
    /* Characteristic Declaration */
    MPU6050_CHAR_ACCEL_Y,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_ACCEL_Y,
    /* Characteristic Declaration */
    MPU6050_CHAR_ACCEL_Z,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_ACCEL_Z,
    /* Characteristic Declaration */
    MPU6050_CHAR_EULER_X,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_EULER_X,
    /* Characteristic Declaration */
    MPU6050_CHAR_EULER_Y,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_EULER_Y,
    /* Characteristic Declaration */
    MPU6050_CHAR_EULER_Z,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_EULER_Z,
    /* Characteristic Declaration */
    MPU6050_CHAR_TEMPERATURE,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_TEMPERATURE,
    // a amazing trick to indicate the length of the enum
    HRS_IDX_NB,
};


void ble_gatts_main(void);