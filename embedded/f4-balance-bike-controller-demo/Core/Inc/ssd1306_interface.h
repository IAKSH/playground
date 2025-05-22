#pragma once
#include <stdint.h>

void ssd1306_delay_ms(uint32_t ms);
uint32_t ssd1306_get_tick(void);

void ssd1306_write_cmd(uint8_t byte);
void ssd1306_write_data(uint8_t* buf,uint16_t len);
