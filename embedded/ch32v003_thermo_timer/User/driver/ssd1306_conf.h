#ifndef __SSD1306_CONF_H__
#define __SSD1306_CONF_H__

#include "ch32v00x.h"

#define SSD1306_DELAY_MS Delay_Ms
#define SSD1306_GET_TICK get_systick

#define SSD1306_I2C_PORT        hi2c1
#define SSD1306_I2C_ADDR        (0x3C << 1)

// Mirror the screen if needed
#define SSD1306_MIRROR_VERT
#define SSD1306_MIRROR_HORIZ

// Set inverse color if needed
// # define SSD1306_INVERSE_COLOR

// Include only needed fonts
#define SSD1306_INCLUDE_FONT_6x8
#define SSD1306_INCLUDE_FONT_7x10
#define SSD1306_INCLUDE_FONT_11x18
#define SSD1306_INCLUDE_FONT_16x26
#define SSD1306_INCLUDE_FONT_16x24
#define SSD1306_INCLUDE_FONT_16x15

// The default value is 128.
#define SSD1306_WIDTH 128
//#define SSD1306_X_OFFSET
// It can be 32, 64 or 128. The default value is 64.
#define SSD1306_HEIGHT 64

#endif /* __SSD1306_CONF_H__ */