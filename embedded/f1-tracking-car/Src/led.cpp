#include "led.hpp"
#include "main.h"

car::Led::Led(TIM_HandleTypeDef& tim,uint8_t channel_r,uint8_t channel_g,uint8_t channel_b)
    : tim(tim),channel_r(channel_r),channel_g(channel_g),channel_b(channel_b) {
    HAL_TIM_PWM_Start(&tim,channel_r);
    HAL_TIM_PWM_Start(&tim,channel_g);
    HAL_TIM_PWM_Start(&tim,channel_b);
}

void car::Led::apply_rgb() {
    __HAL_TIM_SET_COMPARE(&tim,channel_r,r);
    __HAL_TIM_SET_COMPARE(&tim,channel_g,g);
    __HAL_TIM_SET_COMPARE(&tim,channel_b,b);
}