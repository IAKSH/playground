#include "mt9101et.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"

#define DEFAULT_VREF 1100 // 默认参考电压，单位为mV
#define NO_OF_SAMPLES 64 // 采样次数
#define CHANNEL ADC2_CHANNEL_1 // ADC通道

static esp_adc_cal_characteristics_t *adc_chars;

void mt9101et_init(void) {
    // 初始化ADC
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc2_config_channel_atten(CHANNEL, ADC_ATTEN_DB_12);

    // 校准ADC
    adc_chars = calloc(1, sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_characterize(ADC_UNIT_2, ADC_ATTEN_DB_12, ADC_WIDTH_BIT_12, DEFAULT_VREF, adc_chars);
}

void mt9101et_read(uint32_t* raw,uint32_t* voltage) {
    // 读取ADC值
    uint32_t adc_reading = 0;
    for (int i = 0; i < NO_OF_SAMPLES; i++) {
        int raw;
        adc2_get_raw((adc2_channel_t)CHANNEL, ADC_WIDTH_BIT_12, &raw);
        adc_reading += raw;
    }
    adc_reading /= NO_OF_SAMPLES;

    if(raw) {
        *raw = adc_reading;
    }
    if(voltage) {
        *voltage = esp_adc_cal_raw_to_voltage(adc_reading, adc_chars);
    }
}

void mt9101et_kalman_init(Mt9101etKalmanState* kalman_state) {
    kalman_init(&kalman_state->raw, 0.1, 0.1, 0.1, 0);
    kalman_init(&kalman_state->volt, 0.1, 0.1, 0.1, 0);
}

void mt9101et_kalman_update(Mt9101etKalmanState* kalman_state, uint32_t* raw,uint32_t* volt) {
    uint32_t mesure_raw,mesure_volt;

    mt9101et_read(&mesure_raw,&mesure_volt);

    // 使用卡尔曼滤波器更新欧拉角
    *raw = kalman_update(&kalman_state->raw, mesure_raw);
    *volt = kalman_update(&kalman_state->volt, mesure_volt);
}