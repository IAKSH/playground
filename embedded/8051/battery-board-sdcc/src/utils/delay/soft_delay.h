#ifndef __SOFT_DELAY_H__
#define __SOFT_DELAY_H__

#include <stdint.h>

// STC8G1K08A @24MHz 1T

void soft_delay_1sec(void);
void soft_delay_1ms(void);
void soft_delay_10ms(void);
void soft_delay_100ms(void);
void soft_delay_1us(void);
void soft_delay_10us(void);
void soft_delay_100us(void);

#endif