#pragma once

#include <mcs51/8051.h>
#include <stdbool.h>

#define FRONT_DETECTOR_PIN P2_2

inline static bool FD_check(void)
{
    return FRONT_DETECTOR_PIN;
}