#pragma once

#include "driver/i2c.h"

bool qmc5883l_init(i2c_port_t i2c_num);
void qmc5883l_get_data(i2c_port_t i2c_num, float* mag);
float qmc5883l_get_yaw(i2c_port_t i2c_num);