#include <drivers/i2c/i2c.h>
#include "ssd1306.h"
#include "ssd1306_fonts.h"

static void ssd1306_write_command(char command) {
    i2c_start();
    i2c_send_data(0x78);
    i2c_recv_ack();
    i2c_send_data(0x00);  // 控制字节：后续 1 字节为命令
    i2c_recv_ack();
    i2c_send_data(command);
    i2c_recv_ack();
    i2c_stop();
}

static void ssd1306_write_data(char data) {
    i2c_start();
    i2c_send_data(0x78);
    i2c_recv_ack();
    i2c_send_data(0x40);  // 控制字节：后续 1 字节为数据
    i2c_recv_ack();
    i2c_send_data(data);
    i2c_recv_ack();
    i2c_stop();
}

static void ssd1306_setpos(uint8_t x, uint8_t y) {
    ssd1306_write_command(0xB0 + y);
    ssd1306_write_command(((x & 0xF0) >> 4) | 0x10);
    ssd1306_write_command(x & 0x0F);
}

static unsigned int __pow(uint8_t m, uint8_t n) {
    unsigned int result = 1;
    while(n--) result *= m;
    return result;
}

static int __round(float num) {
    return (num >= 0) ? (int)(num + 0.5) : (int)(num - 0.5);
}

void ssd1306_clear(void) {
    uint8_t page, col;
    for(page = 0; page < 4; page++) {
        ssd1306_write_command(0xB0 + page);
        ssd1306_write_command(0x00);
        ssd1306_write_command(0x10);
        
        i2c_start();
        i2c_send_data(0x78);
        i2c_recv_ack();
        i2c_send_data(0x40);
        i2c_recv_ack();
        for(col = 0; col < 128; col++){
            i2c_send_data(0);
            i2c_recv_ack();
        }
        i2c_stop();
    }
}

void ssd1306_draw_char(uint8_t x, uint8_t y, uint8_t chr, uint8_t font_size) {
    uint8_t c, i;
    c = chr - ' ';  // 计算字符在字模数组中的偏移

    if(x > 128 - 1) {
        x = 0;
        y += 2;
    }
    
    if(font_size == 16) {
        ssd1306_setpos(x, y);
        for(i = 0; i < 8; i++)
            ssd1306_write_data(SSD1306_FONT_8X16[c][i + 8]);
        ssd1306_setpos(x, y + 1);
        for(i = 0; i < 8; i++)
            ssd1306_write_data(SSD1306_FONT_8X16[c][i]);
    } else {
        ssd1306_setpos(x, y);
        for(i = 0; i < 6; i++)
            ssd1306_write_data(SSD1306_FONT_6X8[c][i]);
    }
}

void ssd1306_draw_str(uint8_t x, uint8_t y, uint8_t *chr, uint8_t font_size) {
    uint8_t j = 0;
    while (chr[j] != '\0') {
        ssd1306_draw_char(x, y, chr[j], font_size);
        x += 8;
        if(x > 120) {  // 自动换行
            x = 0;
            y += 2;
        }
        j++;
    }
}

void ssd1306_draw_num(uint8_t x, uint8_t y, uint16_t num, uint8_t len, uint8_t num_size) {
    uint8_t pos, digit;
    unsigned int divisor = __pow(10, len - 1);
    for(pos = 0; pos < len; pos++) {
        digit = (uint8_t)((num / divisor) % 10);
        if((num / divisor) == 0 && pos < len - 1)
            ssd1306_draw_char(x + (num_size == 8 ? (num_size / 2 + 2) * pos : (num_size / 2) * pos), y, ' ', num_size);
        else
            ssd1306_draw_char(x + (num_size == 8 ? (num_size / 2 + 2) * pos : (num_size / 2) * pos), y, digit + '0', num_size);
        divisor /= 10;
    }
}

/* SSD1306 初始化 */
void ssd1306_init(void) {    
    ssd1306_write_command(0xAE);
    ssd1306_write_command(0xD5);
    ssd1306_write_command(0x80);
    ssd1306_write_command(0xA8);
    ssd1306_write_command(0x1F);
    ssd1306_write_command(0xD3);
    ssd1306_write_command(0x00);
    
    ssd1306_write_command(0x40);
    
    ssd1306_write_command(0x8D);
    ssd1306_write_command(0x14);
    
    ssd1306_write_command(0x20);
    ssd1306_write_command(0x02);
    ssd1306_write_command(0xA1);
    ssd1306_write_command(0xC8);
    
    ssd1306_write_command(0xDA);
    ssd1306_write_command(0x02);
    
    ssd1306_write_command(0x81);
    ssd1306_write_command(0x8F);
    
    ssd1306_write_command(0xD9);
    ssd1306_write_command(0xF1);
    ssd1306_write_command(0xDB);
    ssd1306_write_command(0x40);
    
    ssd1306_write_command(0xA4);
    ssd1306_write_command(0xA6);
    
    ssd1306_write_command(0x2E);
    
    ssd1306_write_command(0xAF);
}

void ssd1306_draw_float(uint8_t x, uint8_t y, float num, uint8_t len, uint8_t num_size) {
    unsigned int int_part = (unsigned int)num;
    float frac_part = num - int_part;
    unsigned int divisor;
    uint8_t digits = 0;
    unsigned int temp = int_part;

    /* 绘制整数部分 */
    if(temp == 0) {
        ssd1306_draw_char(x, y, '0', num_size);
        x += (num_size == 8 ? (num_size / 2 + 2) : (num_size / 2));
    } else {
        /* 计算整数部分的位数 */
        while (temp) {
            digits++;
            temp /= 10;
        }
        divisor = __pow(10, digits - 1);
        while (divisor) {
            uint8_t d = (uint8_t)(int_part / divisor);
            ssd1306_draw_char(x, y, d + '0', num_size);
            x += (num_size == 8 ? (num_size / 2 + 2) : (num_size / 2));
            int_part %= divisor;
            divisor /= 10;
        }
    }
    
    /* 绘制小数点 */
    ssd1306_draw_char(x, y, '.', num_size);
    x += (num_size == 8 ? (num_size / 2 + 2) : (num_size / 2));
    
    /* 绘制小数部分，逐位转换，不使用任何缓冲数组 */
    {
        uint8_t i;
        for(i = 0; i < len; i++) {
            frac_part *= 10;
            uint8_t d = (uint8_t)frac_part;
            ssd1306_draw_char(x, y, d + '0', num_size);
            x += (num_size == 8 ? (num_size / 2 + 2) : (num_size / 2));
            frac_part -= d;
        }
    }
}

void ssd1306_draw_point(uint8_t x, uint8_t y) {
    uint8_t page = y / 8;              // 每页 8 像素
    uint8_t data = 1 << (y % 8);         // 对应位作为点
    ssd1306_setpos(x, page);           // 设置列和页地址
    ssd1306_write_data(data);          // 直接写数据，覆盖原值
}

void ssd1306_display_off(void) {
    ssd1306_write_command(0xAE);  // 0xAE 为关闭显示命令
}

void ssd1306_display_on(void) {
    ssd1306_write_command(0xAF);  // 0xAF 为打开显示命令
}

void ssd1306_clear_point(uint8_t x, uint8_t y) {
    uint8_t page = y / 8;  // 每页 8 像素
    ssd1306_setpos(x, page);
    ssd1306_write_data(0x00);  // 写入 0 清除该字节（所在列）内所有像素
}
