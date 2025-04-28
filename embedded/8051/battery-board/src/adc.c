#include <intrins.h>
#include "stc8g.h"

#define ADC_Power 0X82
#define ADC_Start 0X40
#define ADC_Flag  0X20
#define ADC_SYS   0X0F

#define VREF 5.0

void adc_init() {
    ADC_CONTR = ADC_Power;
    ADCCFG = ADC_SYS;
}

unsigned short adc_convert() {
    ADC_CONTR |= ADC_Start;
    _nop_();
    _nop_();
    while (!(ADC_CONTR & ADC_Flag));
    ADC_CONTR &= ~ADC_Flag;
    return ADC_RES;
}

float adc_to_voltage(unsigned short adc_value) {
    return (float)adc_value * VREF / 255.0;
}