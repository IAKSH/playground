#pragma once

#include <stdint.h>

extern uint16_t motor_test_duty;

void motor_ledc_initialize(void);
void motor_ledc_test(int count);
void set_motor_duty(uint8_t channel,uint16_t duty);