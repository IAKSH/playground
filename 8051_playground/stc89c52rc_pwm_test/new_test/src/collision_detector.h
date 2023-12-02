#pragma once

#include <mcs51/8051.h>
#include <stdbool.h>

#define CD_COM_PIN P3_7
//#define CD_NO_PIN P0_0        
#define CD_NC_PIN P0_1

inline static void CD_init(void)
{
    CD_NC_PIN = 0;
    //CD_NO_PIN = 0; 
    CD_COM_PIN = 1;
}

inline static bool CD_check(void)
{
    return !CD_NC_PIN;
}