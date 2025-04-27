#pragma once

/* Attributes State Machine */
enum
{
    // Service Declaration
    IDX_SVC,
    // MPU6050
    /* Characteristic Declaration */
    MPU6050_CHAR_ACCEL,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_ACCEL,
    
    /* Characteristic Declaration */
    MPU6050_CHAR_EULER,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_EULER,
    
    /* Characteristic Declaration */
    MPU6050_CHAR_TEMPERATURE,
    /* Characteristic Value */
    MPU6050_CHAR_VAL_TEMPERATURE,
    
    // a amazing trick to indicate the length of the enum
    HRS_IDX_NB,
};


void ble_gatts_main(void);