#include "drone_status.h"

float drone_gryo_euler[3];
float drone_gryo_accel[3];
float drone_gyro_temperature; 

float drone_barometer_pressure;
float drone_barometer_temperature;
float drone_barometer_altitude;

uint16_t drone_motor_duty[5];

uint16_t drone_motor_offset = 2048;