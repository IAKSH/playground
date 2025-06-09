#pragma once
#include <cstdint>
#include <array>
#include "main.h"

namespace car {
    struct Led {
    private:
        TIM_HandleTypeDef& tim;
        uint8_t channel_r,channel_g,channel_b;

    public:
        Led(TIM_HandleTypeDef& tim,uint8_t channel_r,uint8_t channel_g,uint8_t channel_b);
        void apply_rgb();
        std::array<uint16_t,3> rgb{ 0 };
        uint16_t& r{ rgb[0] };
        uint16_t& g{ rgb[1] };
        uint16_t& b{ rgb[2] };
    };
}