#pragma once

#include <stdint.h>
#include <stdbool.h>

#define SSD1306_USE_FRAMEBUFFER 1

#define SSD1306_SCREEN_WIDTH 128
#define SSD1306_SCREEN_HEIGHT 32
#define SSD1306_MIRROR_VERT false
#define SSD1306_MIRROR_HORIZ false

#define SSD1306_ENABLE_FONT_6x8
#define SSD1306_ENABLE_FONT_7x10
#define SSD1306_ENABLE_FONT_11x18
#define SSD1306_ENABLE_FONT_16x15
#define SSD1306_ENABLE_FONT_16x24
#define SSD1306_ENABLE_FONT_16x26

typedef struct {
    const uint8_t w,h;
    const uint16_t *const data;
    const uint8_t *const char_width;
} SSD1306Font;

#ifdef SSD1306_ENABLE_FONT_6x8
extern const SSD1306Font SSD1306Font_6x8;
#endif

#ifdef SSD1306_ENABLE_FONT_7x10
extern const SSD1306Font SSD1306Font_7x10;
#endif

#ifdef SSD1306_ENABLE_FONT_11x18
extern const SSD1306Font SSD1306Font_11x18;
#endif

#ifdef SSD1306_ENABLE_FONT_16x15
extern const SSD1306Font SSD1306Font_16x15;
#endif

#ifdef SSD1306_ENABLE_FONT_16x24
extern const SSD1306Font SSD1306Font_16x24;
#endif

#ifdef SSD1306_ENABLE_FONT_16x26
extern const SSD1306Font SSD1306Font_16x26;
#endif

typedef enum {
    SSD1306_COLOR_WHITE,
    SSD1306_COLOR_BLACK
} SSD1306Color;

typedef struct {
    struct {
        uint8_t x,y;
    } __cursor;
    SSD1306Color color;
    const SSD1306Font* font;
} SSD1306State;

void ssd1306_init(void);
void ssd1306_reset(void);
void ssd1306_create_state(SSD1306State* state); 
void ssd1306_turn_display(bool on);

void ssd1306_set_cursor(SSD1306State* state,uint8_t x,uint8_t y);

void ssd1306_fill(SSD1306State* state,SSD1306Color color);
void ssd1306_flush(SSD1306State* state);

void ssd1306_write_char(SSD1306State* state,char c);
void ssd1306_write_string(SSD1306State* state,const char* s);
void ssd1306_write_int(SSD1306State* state,int i);
void ssd1306_write_float(SSD1306State* state, float f);
void ssd1306_write_float_with_precision(SSD1306State* state,float f,uint8_t precision);

void ssd1306_draw_pixel(SSD1306State* state);
void ssd1306_draw_line(SSD1306State* state,uint8_t x,uint8_t y);
void ssd1306_draw_arc(SSD1306State* state,uint8_t radius,uint16_t start_angle,uint16_t sweep);
void ssd1306_draw_arc_with_radius_line(SSD1306State* state,uint8_t radius,uint16_t start_angle,uint16_t sweep);

void ssd1306_draw_circle(SSD1306State* state,uint8_t r);
void ssd1306_fill_circle(SSD1306State* state,uint8_t r);
void ssd1306_draw_rect(SSD1306State* state,uint8_t x,uint8_t y);
void ssd1306_fill_rect(SSD1306State* state,uint8_t x,uint8_t y);

typedef struct {
    uint8_t x,y;
} SSD1306Vertex2D;

void ssd1306_draw_2d_polyline(SSD1306State* state,const SSD1306Vertex2D* const vertices,uint16_t vertices_len);

void ssd1306_draw_bitmap(SSD1306State* state,uint8_t w,uint8_t h,const uint8_t* bitmap);