#include "battery.hpp"

//car::Battery::Battery(ADC_HandleTypeDef& adc)
//    : adc(adc) {}

float car::get_battery_volt() {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1,50);
    float val = HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    return val * 3.3 / 4096 * 4;
}