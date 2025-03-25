#ifndef __SSD1306_H__
#define __SSD1306_H__

#include <stdint.h>

void ssd1306_init(void);
void ssd1306_clear(void);
void ssd1306_draw_char(uint8_t x, uint8_t y, uint8_t chr, uint8_t font_size);
void ssd1306_draw_str(uint8_t x, uint8_t y, uint8_t *str, uint8_t font_size);
void ssd1306_draw_num(uint8_t x, uint8_t y, uint16_t num, uint8_t len, uint8_t num_size);
void ssd1306_draw_float(uint8_t x, uint8_t y, float num, uint8_t len, uint8_t num_size);

void ssd1306_draw_point(uint8_t x,uint8_t y);
void ssd1306_clear_point(uint8_t x, uint8_t y);

void ssd1306_display_off(void);
void ssd1306_display_on(void);

#endif /* __SSD1306_H__ */
