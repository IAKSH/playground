#pragma once
#include <cstdint>
#include <array>
#include "main.h"

namespace car {
    struct HBridgeConfig {
        TIM_HandleTypeDef& tim;
        uint16_t channel_a;
        uint16_t channel_b;
        GPIO_TypeDef* const control_port_a1;
        GPIO_TypeDef* const control_port_a2;
        GPIO_TypeDef* const control_port_b1;
        GPIO_TypeDef* const control_port_b2;
        uint16_t control_pin_a1;
        uint16_t control_pin_a2;
        uint16_t control_pin_b1;
        uint16_t control_pin_b2;
    };

    class HBridge {
    private:
        HBridgeConfig config;
        bool clutch;

    public:
        std::array<int,2> speed;
        explicit HBridge(const HBridgeConfig& config);
        void apply_speed();
    };
}
