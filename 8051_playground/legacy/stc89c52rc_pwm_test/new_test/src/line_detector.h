#pragma once

#include <mcs51/8051.h>
#include <stdbool.h>

#define LD_PIN_1 P0_2
#define LD_PIN_2 P0_3
#define LD_PIN_3 P0_4
#define LD_PIN_4 P0_5

inline static bool LD_check_1(void)
{
    return !LD_PIN_1;
}

inline static bool LD_check_2(void)
{
    return !LD_PIN_2;
}

inline static bool LD_check_3(void)
{
    return !LD_PIN_3;
}

inline static bool LD_check_4(void)
{
    return !LD_PIN_4;
}