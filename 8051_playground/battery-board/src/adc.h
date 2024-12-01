#ifndef __ADC_H__
#define __ADC_H__

void adc_init();
unsigned short adc_convert();
float adc_to_voltage(unsigned short adc_value);

#endif
