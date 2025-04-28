#pragma once

#include <mcs51/8051.h>
#include <stdbool.h>

#define SD_RIGHT_PIN P3_2 // INT0
#define SD_LEFT_PIN P3_3 // INT1

inline static bool __sd_left;
inline static bool __sd_right;
inline static int __sd_interrupt_count;

inline static void SD_reload(void)
{
    __sd_left = false;
    __sd_right = false;

    EA = 1;
    EX0 = 1;
    EX1 = 1;
    TR1 = 0;
}

// 1ms @11.0592MHz
inline static void SD_init_timer1(void)
{
	PCON |= 0x40;
	TMOD &= 0x0F;
	TL1 = 0xCD;
	TH1 = 0xD4;
	TF1 = 0;
    EA = 1;
    ET1 = 1;
    TR1 = 0;
}

inline static void SD_interrupt(void) __interrupt(3)
{
    TR1 = 0;
    TL1 = 0xCD;
	TH1 = 0xD4;
    TR1 = 1;

    ++__sd_interrupt_count;

    // SD触发100ms之后清除SD状态
    if(__sd_interrupt_count >= 500)
    {
        TR1 = 0;
        SD_reload();
        __sd_interrupt_count = 0;
    }
}

inline static void SD_init(void)
{
    // 初始化timer1，用于定时重载SD
    __sd_interrupt_count = 0;
    SD_init_timer1();
    // 先重载一次
    SD_reload();
}

// 外部中断
inline static void update_sd_right(void) __interrupt(0)
{
    __sd_right = true;
    IE0 = 0;
    EX0 = 0;
    TR1 = 1;
}

// 外部中断
inline static void update_sd_left(void) __interrupt(2)
{
    __sd_left = true;
    IE1 = 0;
    EX1 = 0;
    TR1 = 1;
}

inline static bool SD_check_ever_right(void)
{
    return __sd_right;
}

inline static bool SD_check_ever_left(void)
{
    return __sd_left;
}