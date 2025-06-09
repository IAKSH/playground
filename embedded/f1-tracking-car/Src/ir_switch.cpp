#include "ir_switch.hpp"

car::IRSwitch::IRSwitch(GPIO_TypeDef* port,uint16_t pin)
    : gpio_port(port),gpio_pin(pin) {}

bool car::IRSwitch::activated() {
    bool state = HAL_GPIO_ReadPin(gpio_port,gpio_pin);
    if(last_state != state) {
        last_state = state;
        last_change_tick = osKernelGetTickCount();
    }
    return state;
}

uint32_t car::IRSwitch::get_last_change_tick() const {
    return last_change_tick;
}
