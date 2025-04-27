#pragma once
#include <stdint.h>

//extern vec3_t drone_magnetic_angle;
//extern float drone_battery_volte;

extern float drone_gryo_euler[3];
extern float drone_gryo_accel[3];
extern float drone_gyro_temperature; 
 
extern float drone_barometer_pressure;
extern float drone_barometer_temperature;
extern float drone_barometer_altitude;

extern uint16_t drone_motor_duty[5];

// for height
extern uint16_t drone_motor_offset;
extern float drone_target_euler[3];

extern float drone_motor_kp;
extern float drone_motor_ki;
extern float drone_motor_kd;

extern uint8_t drone_motor_emergency_stop;