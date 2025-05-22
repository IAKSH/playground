#include "ssd1306_interface.h"
#include "main.h"

#define SSD1306_I2C_ADDR (0x3C << 1)

void ssd1306_delay_ms(uint32_t ms) {
    HAL_Delay(ms);
}

uint32_t ssd1306_get_tick(void) {
    return HAL_GetTick();
}

void ssd1306_write_cmd(uint8_t byte) {
    HAL_I2C_Mem_Write(&hi2c1, SSD1306_I2C_ADDR, 0x00, 1, &byte, 1, HAL_MAX_DELAY);
}

void ssd1306_write_data(uint8_t* buf,uint16_t len) {
    HAL_I2C_Mem_Write(&hi2c1, SSD1306_I2C_ADDR, 0x40, 1, buf, len, HAL_MAX_DELAY);
}