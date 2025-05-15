#pragma once

#include <stdint.h>
#include <stdbool.h>

void i2c_setup(void);
void i2c_write(uint8_t addr,uint8_t *buf,uint16_t len);
void i2c_write_reg(uint8_t addr,uint8_t reg_addr,uint8_t* data,uint16_t size);
void i2c_read(uint8_t addr, uint8_t *buf, uint16_t len);