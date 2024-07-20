#include "motor.h"

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_err.h"

#define MOTOR_LEDC_HS_TIMER          LEDC_TIMER_0
#define MOTOR_LEDC_HS_MODE           LEDC_HIGH_SPEED_MODE
#define MOTOR_LEDC_HS_CH0_GPIO       (18)
#define MOTOR_LEDC_HS_CH0_CHANNEL    LEDC_CHANNEL_0
#define MOTOR_LEDC_HS_CH1_GPIO       (4)
#define MOTOR_LEDC_HS_CH1_CHANNEL    LEDC_CHANNEL_1
#define MOTOR_LEDC_HS_CH2_GPIO       (5)
#define MOTOR_LEDC_HS_CH2_CHANNEL    LEDC_CHANNEL_2
#define MOTOR_LEDC_HS_CH3_GPIO       (6)
#define MOTOR_LEDC_HS_CH3_CHANNEL    LEDC_CHANNEL_3
#define MOTOR_LEDC_HS_CH4_GPIO       (7)
#define MOTOR_LEDC_HS_CH4_CHANNEL    LEDC_CHANNEL_4

#define MOTOR_TEST_CH_NUM       (5)
#define MOTOR_TEST_DUTY         (4000)    // 渐变的变大最终目标占空比
#define MOTOR_TEST_FADE_TIME    (3000)    // 变化时长

static ledc_channel_config_t ledc_channel[MOTOR_TEST_CH_NUM] =
{
    {
        .channel    = MOTOR_LEDC_HS_CH0_CHANNEL,
        .duty       = 0,
        .gpio_num   = MOTOR_LEDC_HS_CH0_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint     = 0,
        .timer_sel  = MOTOR_LEDC_HS_TIMER
    },
    {
        .channel    = MOTOR_LEDC_HS_CH1_CHANNEL,
        .duty       = 0,
        .gpio_num   = MOTOR_LEDC_HS_CH1_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint     = 0,
        .timer_sel  = MOTOR_LEDC_HS_TIMER
    },
    {
        .channel    = MOTOR_LEDC_HS_CH2_CHANNEL,
        .duty       = 0,
        .gpio_num   = MOTOR_LEDC_HS_CH2_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint     = 0,
        .timer_sel  = MOTOR_LEDC_HS_TIMER
    },
    {
        .channel    = MOTOR_LEDC_HS_CH3_CHANNEL,
        .duty       = 0,
        .gpio_num   = MOTOR_LEDC_HS_CH3_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint     = 0,
        .timer_sel  = MOTOR_LEDC_HS_TIMER
    },
    {
        .channel    = MOTOR_LEDC_HS_CH4_CHANNEL,
        .duty       = 0,
        .gpio_num   = MOTOR_LEDC_HS_CH4_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint     = 0,
        .timer_sel  = MOTOR_LEDC_HS_TIMER
    }
};

static ledc_timer_config_t ledc_timer =
{
    .duty_resolution = LEDC_TIMER_13_BIT,   // PWM占空比分辨率
    .freq_hz = 5000,                        // PWM信号频率
    .speed_mode = LEDC_LOW_SPEED_MODE,      // 定时器模式
    .timer_num = MOTOR_LEDC_HS_TIMER,       // 定时器序号
    .clk_cfg = LEDC_AUTO_CLK,               // Auto select the source clock
};

void motor_ledc_initialize(void) {
    /*
     * Prepare and set configuration of timers
     * that will be used by LED Controller
     */
    
    // Set configuration of timer0 for high speed channels
    ledc_timer_config(&ledc_timer);

    /*
     * Prepare individual configuration
     * for each channel of LED Controller
     * by selecting:
     * - controller's channel number
     * - output duty cycle, set initially to 0
     * - GPIO number where LED is connected to
     * - speed mode, either high or low
     * - timer servicing selected channel
     *   Note: if different channels use one timer,
     *         then frequency and bit_num of these channels
     *         will be the same
     */

    // 配置LED控制器
    for (int ch = 0; ch < MOTOR_TEST_CH_NUM; ch++) {
        ledc_channel_config(&ledc_channel[ch]);
    }
}

void motor_ledc_test(int count) {
    // 初始化淡入淡出服务
    ledc_fade_func_install(0);    // 注册LEDC服务，在调用前使用，参数是作为是否允许中断
 
    for(int i = 0;i < count;i++) {
        printf("1. LEDC fade up to duty = %d\n", MOTOR_TEST_DUTY);
        for (int ch = 0; ch < MOTOR_TEST_CH_NUM; ch++) {
            // 配置LEDC定时器
            ledc_set_fade_with_time(ledc_channel[ch].speed_mode,
                    ledc_channel[ch].channel, MOTOR_TEST_DUTY, MOTOR_TEST_FADE_TIME);
            // 开始渐变
            ledc_fade_start(ledc_channel[ch].speed_mode,
                    ledc_channel[ch].channel, LEDC_FADE_NO_WAIT);
        }
        // 等待渐变完成
        vTaskDelay(pdMS_TO_TICKS(MOTOR_TEST_FADE_TIME));

        printf("2. LEDC fade down to duty = 0\n");
        for (int ch = 0; ch < MOTOR_TEST_CH_NUM; ch++) {
            ledc_set_fade_with_time(ledc_channel[ch].speed_mode,
                    ledc_channel[ch].channel, 0, MOTOR_TEST_FADE_TIME);
            ledc_fade_start(ledc_channel[ch].speed_mode,
                    ledc_channel[ch].channel, LEDC_FADE_NO_WAIT);
        }
        vTaskDelay(pdMS_TO_TICKS(MOTOR_TEST_FADE_TIME));

        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ledc_fade_func_uninstall();
}