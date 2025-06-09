#pragma once
#include "main.h"

namespace car {
    class IRSwitch {
    private:
        GPIO_TypeDef* const gpio_port;
        uint16_t gpio_pin;
        bool last_state = false;
        uint32_t last_change_tick = 0;
    public:
        IRSwitch(GPIO_TypeDef* port,uint16_t pin);
        bool activated();
        uint32_t get_last_change_tick() const;
    };
}