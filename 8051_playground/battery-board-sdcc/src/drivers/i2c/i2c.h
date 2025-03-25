#ifndef __I2C_H__
#define __I2C_H__

#include <stdint.h>

void i2c_init(void);
void i2c_wait(void);
void i2c_start(void);
void i2c_send_data(uint8_t data);
void i2c_recv_ack(void);
uint8_t i2c_recv_data(void);
void i2c_send_ack(void);
void i2c_send_nak(void);
void i2c_stop(void);

#endif