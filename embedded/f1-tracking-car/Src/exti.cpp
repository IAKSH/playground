#include "main.h"
#include "status.hpp"

extern "C" {
    void HAL_GPIO_EXTI_Callback(uint16_t);
}

void HAL_GPIO_EXTI_Callback(uint16_t gpio_pin) {
    switch(gpio_pin) {
    case ULTRA_SONIC_ECHO_Pin:
        if(HAL_GPIO_ReadPin(ULTRA_SONIC_ECHO_GPIO_Port, ULTRA_SONIC_ECHO_Pin) == GPIO_PIN_SET) {
            // rising
            ultra_sonic.irq_rising();
        } else {
            // falling
            ultra_sonic.irq_falling();
        }
        break;
    default:
        break;
    }
}