#ifndef __UART_H__
#define __UART_H__

#include <stdint.h>

void uart_init(void);
void uart_send_byte(uint8_t byte);
void uart_send_string(const char *str);
void uart_send_uint16(uint16_t num);
void uart_send_float(float num);

#endif