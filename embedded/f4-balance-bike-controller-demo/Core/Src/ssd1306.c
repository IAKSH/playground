#include "ssd1306.h"
#include "ssd1306_interface.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if SSD1306_USE_FRAMEBUFFER
#define SSD1306_FRAMEBUFFER_SIZE (SSD1306_SCREEN_WIDTH * (SSD1306_SCREEN_HEIGHT / 8))
static uint8_t framebuffer[SSD1306_FRAMEBUFFER_SIZE];
#else
// 非framebuffer模式下只缓存一页（1页=SSD1306_SCREEN_WIDTH字节）
static uint8_t page_buffer[SSD1306_SCREEN_WIDTH];
static uint8_t page_buffer_index = 0xFF; // 当前缓存的页号，0xFF表示无效
#endif

static void ftoa(float f, char* buf, unsigned char precision) {
    char *p = buf;
    
    if (f < 0) {
        *p++ = '-';
        f = -f;
    }
    
    int ipart = (int)f;
    float fpart = f - (float)ipart;
    itoa(ipart, p, 10);
    while (*p)
        p++;
    
    *p++ = '.';

    int pow10 = 1;
    for (unsigned char i = 0; i < precision; i++) {
        pow10 *= 10;
    }

    int f_int = (int)(fpart * pow10 + 0.5f);
    
    char frac[10] = {0};
    itoa(f_int, frac, 10);
    
    int frac_len = 0;
    while (frac[frac_len]) {
        frac_len++;
    }
    for (unsigned char i = 0; i < precision - frac_len; i++) {
        *p++ = '0';
    }
    strcpy(p, frac);
}

static inline float deg2rad(float deg) {
    return deg * (M_PI / 180.0f);
}

static inline uint16_t normalize_angle(uint16_t angle) {
    return angle % 360;
}

#if SSD1306_USE_FRAMEBUFFER
// 统一像素绘制，仅支持framebuffer
static inline void draw_pixel(int x, int y, SSD1306Color color) {
    if ((unsigned)x >= SSD1306_SCREEN_WIDTH || (unsigned)y >= SSD1306_SCREEN_HEIGHT) return;
    uint16_t idx = x + (y / 8) * SSD1306_SCREEN_WIDTH;
    uint8_t bit = 1 << (y % 8);
    if (color == SSD1306_COLOR_WHITE) framebuffer[idx] |= bit;
    else framebuffer[idx] &= ~bit;
}

// 通用线段绘制
static void draw_line_raw(int x0, int y0, int x1, int y1, SSD1306Color color) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1, err = dx - dy;
    while (1) {
        draw_pixel(x0, y0, color);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}
#else
// 非framebuffer模式下的像素绘制，带页缓存（仅缓存一页）
static inline void draw_pixel(int x, int y, SSD1306Color color) {
    if ((unsigned)x >= SSD1306_SCREEN_WIDTH || (unsigned)y >= SSD1306_SCREEN_HEIGHT) return;
    uint8_t page = y / 8;
    uint8_t bit = 1 << (y % 8);

    // 如果缓存页不是当前页，先刷新缓存
    if (page_buffer_index != page) {
        // 可选：如有读功能可从屏幕读，否则只能假设全0
        memset(page_buffer, 0, SSD1306_SCREEN_WIDTH);
        page_buffer_index = page;
    }

    // 修改缓存
    if (color == SSD1306_COLOR_WHITE)
        page_buffer[x] |= bit;
    else
        page_buffer[x] &= ~bit;

    // 写回SSD1306对应页的单字节
    ssd1306_write_cmd(0xB0 | page); // 页地址
    ssd1306_write_cmd(0x00 | (x & 0x0F)); // 低列地址
    ssd1306_write_cmd(0x10 | (x >> 4));   // 高列地址
    uint8_t data = page_buffer[x];
    ssd1306_write_data(&data, 1);
}

// 通用线段绘制
static void draw_line_raw(int x0, int y0, int x1, int y1, SSD1306Color color) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1, err = dx - dy;
    while (1) {
        draw_pixel(x0, y0, color);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}
#endif

void ssd1306_reset(void) {
}

// framebuffer中的1bit表示一个像素
void ssd1306_create_framebuffer(void) {
#if SSD1306_USE_FRAMEBUFFER
    memset(framebuffer, 0, SSD1306_FRAMEBUFFER_SIZE);
#else
    // 非framebuffer模式清空页缓存
    memset(page_buffer, 0, SSD1306_SCREEN_WIDTH);
    page_buffer_index = 0xFF;
#endif
}

void ssd1306_create_state(SSD1306State* state) {
    state->__cursor.x = 0;
    state->__cursor.y = 0;
    state->color = SSD1306_COLOR_WHITE;
    state->font = NULL;
}

void ssd1306_turn_display(bool on) {
    ssd1306_write_cmd(on ? 0xAF : 0xAE);
}

void ssd1306_init(void) {
    ssd1306_reset();
    ssd1306_delay_ms(100);

    ssd1306_turn_display(false);

    ssd1306_write_cmd(0x20); // Set Memory Addressing Mode
    ssd1306_write_cmd(0x00); // Horizontal Addressing Mode

    // 可选：设置页起始地址
    ssd1306_write_cmd(0xB0);

    if (SSD1306_MIRROR_VERT)
        ssd1306_write_cmd(0xC0);
    else
        ssd1306_write_cmd(0xC8);

    ssd1306_write_cmd(0x00); // set low column address
    ssd1306_write_cmd(0x10); // set high column address

    // 设置start line address
    ssd1306_write_cmd(0x40 | 0x00); // 通常为0

    ssd1306_write_cmd(0x81); // set contrast
    ssd1306_write_cmd(0xFF);

    if (SSD1306_MIRROR_HORIZ)
        ssd1306_write_cmd(0xA0);
    else
        ssd1306_write_cmd(0xA1);

    ssd1306_write_cmd(0xA6); // normal display

    // 自动设置multiplex ratio
    ssd1306_write_cmd(0xA8);
    ssd1306_write_cmd(SSD1306_SCREEN_HEIGHT - 1);

    ssd1306_write_cmd(0xA4); // Output follows RAM content

    ssd1306_write_cmd(0xD3); // display offset
    ssd1306_write_cmd(0x00);

    ssd1306_write_cmd(0xD5); // display clock divide ratio/oscillator frequency
    ssd1306_write_cmd(0xF0);

    ssd1306_write_cmd(0xD9); // pre-charge period
    ssd1306_write_cmd(0x22);

    // 自动设置com pins hardware configuration
    ssd1306_write_cmd(0xDA);
    if (SSD1306_SCREEN_HEIGHT == 32)
        ssd1306_write_cmd(0x02);
    else
        ssd1306_write_cmd(0x12);

    ssd1306_write_cmd(0xDB); // vcomh
    ssd1306_write_cmd(0x20);

    ssd1306_write_cmd(0x8D); // DC-DC enable
    ssd1306_write_cmd(0x14);

    ssd1306_turn_display(true);
}

void ssd1306_flush(SSD1306State* state) {
#if SSD1306_USE_FRAMEBUFFER
    if (!state)
        return;
    for (uint8_t page = 0; page < (SSD1306_SCREEN_HEIGHT / 8); page++) {
        ssd1306_write_cmd(0xB0 | page); // 设置页地址
        ssd1306_write_cmd(0x00);        // 设置低列地址
        ssd1306_write_cmd(0x10);        // 设置高列地址
        uint16_t offset = page * SSD1306_SCREEN_WIDTH;
        ssd1306_write_data(&framebuffer[offset], SSD1306_SCREEN_WIDTH);
    }
#else
    // 非framebuffer模式flush无实际意义，仅清空页缓存
    page_buffer_index = 0xFF;
#endif
}

void ssd1306_write_char(SSD1306State* state, char c) {
    if (c < 32 || c > 126)
        return;
#ifdef SSD1306_ENABLE_FONT_6x8
    const SSD1306Font* defaultFont = &SSD1306Font_6x8;
#else
    const SSD1306Font* defaultFont = NULL;
#endif
    const SSD1306Font* font = state->font ? state->font : defaultFont;
    if (!font)
        return;
    uint8_t char_width = font->char_width ? font->char_width[c - 32] : font->w;
    for (uint8_t row = 0; row < font->h; row++) {
        uint16_t row_data = font->data[(c - 32) * font->h + row];
        for (uint8_t col = 0; col < char_width; col++) {
            if ((row_data << col) & 0x8000)
                draw_pixel(state->__cursor.x + col, state->__cursor.y + row, state->color);
            else {
                SSD1306Color bg = (state->color == SSD1306_COLOR_WHITE) ? SSD1306_COLOR_BLACK : SSD1306_COLOR_WHITE;
                draw_pixel(state->__cursor.x + col, state->__cursor.y + row, bg);
            }
        }
    }
    state->__cursor.x += char_width;
}

void ssd1306_write_string(SSD1306State* state, const char* s) {
    while (*s) {
        ssd1306_write_char(state, *s);
        s++;
    }
}

void ssd1306_write_int(SSD1306State* state, int i) {
    char buf[12];
    itoa(i, buf, 10);
    ssd1306_write_string(state, buf);
}

void ssd1306_write_float(SSD1306State* state, float f) {
    char buf[20];
    ftoa(f, buf, 2);
    ssd1306_write_string(state, buf);
}

void ssd1306_write_float_with_precision(SSD1306State* state, float f,uint8_t precision) {
    char buf[20];
    ftoa(f, buf, precision);
    ssd1306_write_string(state, buf);
}

void ssd1306_draw_pixel(SSD1306State* state) {
    draw_pixel(state->__cursor.x, state->__cursor.y, state->color);
}

void ssd1306_draw_line(SSD1306State* state, uint8_t x, uint8_t y) {
    draw_line_raw(state->__cursor.x, state->__cursor.y, x, y, state->color);
    state->__cursor.x = x; state->__cursor.y = y;
}

void ssd1306_draw_arc(SSD1306State* state, uint8_t radius, uint16_t start_angle, uint16_t sweep) {
    int cx = state->__cursor.x, cy = state->__cursor.y;
    uint16_t norm_start = normalize_angle(start_angle);
    uint16_t norm_sweep = normalize_angle(sweep);
    uint8_t segments = (norm_sweep * 36) / 360;
    if (segments < 1) segments = 1;
    float angle_step = (float)norm_sweep / segments;
    float angle = norm_start;
    int prev_x = cx + (int)(sin(deg2rad(angle)) * radius);
    int prev_y = cy + (int)(cos(deg2rad(angle)) * radius);
    for (uint8_t i = 1; i <= segments; i++) {
        angle = norm_start + i * angle_step;
        int curr_x = cx + (int)(sin(deg2rad(angle)) * radius);
        int curr_y = cy + (int)(cos(deg2rad(angle)) * radius);
        draw_line_raw(prev_x, prev_y, curr_x, curr_y, state->color);
        prev_x = curr_x;
        prev_y = curr_y;
    }
}

void ssd1306_draw_arc_with_radius_line(SSD1306State* state, uint8_t radius, uint16_t start_angle, uint16_t sweep) {
    ssd1306_draw_arc(state, radius, start_angle, sweep);
    int cx = state->__cursor.x, cy = state->__cursor.y;
    uint16_t norm_start = normalize_angle(start_angle);
    uint16_t norm_end = normalize_angle(start_angle + sweep);
    int start_x = cx + (int)(sin(deg2rad(norm_start)) * radius);
    int start_y = cy + (int)(cos(deg2rad(norm_start)) * radius);
    int end_x   = cx + (int)(sin(deg2rad(norm_end)) * radius);
    int end_y   = cy + (int)(cos(deg2rad(norm_end)) * radius);
    draw_line_raw(cx, cy, start_x, start_y, state->color);
    draw_line_raw(cx, cy, end_x, end_y, state->color);
}

void ssd1306_draw_circle(SSD1306State* state, uint8_t r) {
    int cx = state->__cursor.x, cy = state->__cursor.y, x = -r, y = 0, err = 2 - 2 * r;
    do {
        draw_pixel(cx - x, cy + y, state->color);
        draw_pixel(cx + x, cy + y, state->color);
        draw_pixel(cx + x, cy - y, state->color);
        draw_pixel(cx - x, cy - y, state->color);
        int e2 = err;
        if (e2 <= y) { y++; err += y * 2 + 1; }
        if (e2 > x) { x++; err += x * 2 + 1; }
    } while (x <= 0);
}

void ssd1306_fill_circle(SSD1306State* state, uint8_t r) {
    int cx = state->__cursor.x, cy = state->__cursor.y;
    for (int dy = -r; dy <= r; dy++) {
        int dx = (int)sqrt(r * r - dy * dy);
        for (int x = cx - dx; x <= cx + dx; x++)
            draw_pixel(x, cy + dy, state->color);
    }
}

void ssd1306_draw_rect(SSD1306State* state, uint8_t w, uint8_t h) {
    int x0 = state->__cursor.x, y0 = state->__cursor.y, x1 = x0 + w - 1, y1 = y0 + h - 1;
    for (int i = x0; i <= x1; i++) draw_pixel(i, y0, state->color);
    for (int i = x0; i <= x1; i++) draw_pixel(i, y1, state->color);
    for (int j = y0; j <= y1; j++) draw_pixel(x0, j, state->color);
    for (int j = y0; j <= y1; j++) draw_pixel(x1, j, state->color);
}

void ssd1306_fill_rect(SSD1306State* state, uint8_t w, uint8_t h) {
    int x0 = state->__cursor.x, y0 = state->__cursor.y;
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            draw_pixel(x0 + i, y0 + j, state->color);
}

void ssd1306_draw_2d_polyline(SSD1306State* state, const SSD1306Vertex2D* v, uint16_t n) {
    if (!v || n < 2) return;
    for (uint16_t i = 1; i < n; i++)
        draw_line_raw(v[i - 1].x, v[i - 1].y, v[i].x, v[i].y, state->color);
}

void ssd1306_draw_bitmap(SSD1306State* state, uint8_t w, uint8_t h, const uint8_t* bmp) {
    int byteWidth = (w + 7) / 8, x0 = state->__cursor.x, y0 = state->__cursor.y;
    for (uint8_t j = 0; j < h; j++)
        for (uint8_t i = 0; i < w; i++)
            if (bmp[j * byteWidth + i / 8] & (0x80 >> (i % 8)))
                draw_pixel(x0 + i, y0 + j, state->color);
}

void ssd1306_fill(SSD1306State* state, SSD1306Color color) {
#if SSD1306_USE_FRAMEBUFFER
    memset(framebuffer, color == SSD1306_COLOR_WHITE ? 0xFF : 0x00, SSD1306_FRAMEBUFFER_SIZE);
#else
    // 填充每一页并写入屏幕
    uint8_t fill = (color == SSD1306_COLOR_WHITE) ? 0xFF : 0x00;
    for (uint8_t page = 0; page < (SSD1306_SCREEN_HEIGHT / 8); page++) {
        memset(page_buffer, fill, SSD1306_SCREEN_WIDTH);
        page_buffer_index = page;
        ssd1306_write_cmd(0xB0 | page);
        ssd1306_write_cmd(0x00);
        ssd1306_write_cmd(0x10);
        ssd1306_write_data(page_buffer, SSD1306_SCREEN_WIDTH);
    }
    page_buffer_index = 0xFF;
#endif
}

void ssd1306_set_cursor(SSD1306State* state,uint8_t x,uint8_t y) {
    state->__cursor.x = x;
    state->__cursor.y = y;
#if !SSD1306_USE_FRAMEBUFFER
    // 非framebuffer模式下设置硬件光标
    ssd1306_write_cmd(0xB0 | (y / 8));
    ssd1306_write_cmd(0x00 | (x & 0x0F));
    ssd1306_write_cmd(0x10 | (x >> 4));
#endif
}