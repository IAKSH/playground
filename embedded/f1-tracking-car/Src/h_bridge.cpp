#include "h_bridge.hpp"
#include "main.h"
#include <cmath>

#define MOTOR_PWM_TIMER htim2

// timer 2, channel 1 and 4
car::HBridge::HBridge(const HBridgeConfig& config)
    : config(config) {
    speed = {0,0};
    HAL_TIM_PWM_Start(&config.tim,config.channel_a);
    HAL_TIM_PWM_Start(&config.tim,config.channel_b);
    apply_speed();
}

void car::HBridge::apply_speed() {
    // 1. apply speed
    __HAL_TIM_SET_COMPARE(&config.tim,config.channel_a,std::abs(speed[0]));
    __HAL_TIM_SET_COMPARE(&config.tim,config.channel_b,std::abs(speed[1]));

    // 2. change direction
    constexpr GPIO_PinState MOTOR_FORWARD{ GPIO_PIN_SET };
    constexpr GPIO_PinState MOTOR_BACKWARD{ GPIO_PIN_RESET };
    HAL_GPIO_WritePin(config.control_port_a1,config.control_pin_a1,speed[0] < 0 ? MOTOR_FORWARD : MOTOR_BACKWARD);
    HAL_GPIO_WritePin(config.control_port_a2,config.control_pin_a2,speed[0] > 0 ? MOTOR_FORWARD : MOTOR_BACKWARD);
    HAL_GPIO_WritePin(config.control_port_b1,config.control_pin_b1,speed[1] < 0 ? MOTOR_FORWARD : MOTOR_BACKWARD);
    HAL_GPIO_WritePin(config.control_port_b2,config.control_pin_b2,speed[1] > 0 ? MOTOR_FORWARD : MOTOR_BACKWARD);
}
