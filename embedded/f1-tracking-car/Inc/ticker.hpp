#pragma once
#include "main.h"

namespace car {
    class Ticker {
    private:
        TIM_HandleTypeDef* tim;
        uint32_t cnt;

    public:
        Ticker(TIM_HandleTypeDef* tim);
        void start();
        void stop();
        void irq();
        void clear();
        uint32_t get_us() const;
        uint32_t get_cnt() const;
    };
}
