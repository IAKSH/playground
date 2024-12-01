#include "stc8g.h"
#include "oled.h"
#include "i2c.h"
#include "adc.h"

unsigned short adc_result;
float voltage;
unsigned short adc_results[10];

//@11.0592MHz
void Delay1000ms(void) {
    unsigned char data i, j, k;
    i = 57;
    j = 27;
    k = 112;
    do
    {
        do
        {
            while (--k)
                ;
        } while (--j);
    } while (--i);
}

unsigned short get_average_adc(void) {
    unsigned short temp;
    unsigned int sum = 0;
    unsigned char i, j;

    for (i = 0; i < 10; i++) {
        adc_results[i] = adc_convert();
    }

    for (i = 0; i < 10 - 1; i++) {
        for (j = 0; j < 10 - 1 - i; j++)
        {
            if (adc_results[j] > adc_results[j + 1])
            {
                temp = adc_results[j];
                adc_results[j] = adc_results[j + 1];
                adc_results[j + 1] = temp;
            }
        }
    }

    for (i = 1; i < 9; i++) {
        sum += adc_results[i];
    }

    return (sum / 8);
}

void main(void) {
    P3M0 = 0x00;
    P3M1 = 0x00;
    P5M0 = 0x00;
    P5M1 = 0x00;
	
    // P3_2 (ADC2) 设置开漏输出
	P3M0 |= 0x04; P3M1 |= 0x04; 

    // 启用I2C2
    P_SW2 = 0x80;
    P_SW2 &= ~(1 << 5);
    P_SW2 |= (1 << 4);

    I2CCFG = 0xe0;
    I2CMSST = 0x00;
    OLED_Init();
    OLED_Clear();

    adc_init();

    OLED_ShowString(16, 1, "Hello world!", 16);
    Delay1000ms();
    OLED_Clear();

    while (1) {
        adc_result = get_average_adc();
        voltage = adc_to_voltage(adc_result);

        OLED_ShowString(0, 0, "adc: ", 16);
        OLED_ShowNum(38, 0, adc_result, 3, 16);
        OLED_ShowString(0, 2, "vol: ", 16);
        OLED_ShowFloat(38, 2, voltage * 2.38f, 2, 16);
    }
}

