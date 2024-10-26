#include "drone_status.h"

float drone_gryo_euler[3];
float drone_gryo_accel[3];
float drone_gyro_temperature; 

float drone_barometer_pressure;
float drone_barometer_temperature;
float drone_barometer_altitude;

uint16_t drone_motor_duty[5];

uint16_t drone_motor_offset = 2048;
float drone_target_euler[3] = {0,0,0};

float drone_motor_kp = 5.0f;
float drone_motor_ki = 0.1f;
float drone_motor_kd = 1.0f;

uint8_t drone_motor_emergency_stop = 0;