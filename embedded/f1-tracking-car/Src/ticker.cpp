#include "ticker.hpp"

car::Ticker::Ticker(TIM_HandleTypeDef* tim) 
    : tim(tim) {}

void car::Ticker::start() {
    //HAL_TIM_Base_Start(tim);
    __HAL_TIM_CLEAR_IT(tim, TIM_IT_UPDATE);	
    HAL_TIM_Base_Start_IT(tim);
}

void car::Ticker::stop() {
    //HAL_TIM_Base_Stop(tim);
    HAL_TIM_Base_Stop_IT(tim);
}

void car::Ticker::irq() {
    ++cnt;
}

void car::Ticker::clear() {
    //__HAL_TIM_SET_COUNTER(tim,0);
    cnt = 0;
}

uint32_t car::Ticker::get_us() const {
    uint32_t val = __HAL_TIM_GET_COUNTER(tim);
    val += cnt * 65535;
    return val;
}

uint32_t car::Ticker::get_cnt() const {
    return cnt;
}
