#include <cmsis_os2.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include "tasks.h"
#include "led.hpp"
#include "status.hpp"
#include "main.h"

static void rgb_loop(car::Led& led) {
    printf("[led] running RGB\n");
    
    float x = 0;
    constexpr float speed = 0.05f;
    constexpr uint16_t LED_PWM_MAX = 50;
    while(true) {
        // 相差120度
        led.rgb = {
            static_cast<uint16_t>((sinf(x) * 0.5f + 0.5f) * LED_PWM_MAX),
            static_cast<uint16_t>((sinf(x + 2.0944f) * 0.5f + 0.5f) * LED_PWM_MAX), // 2π/3 ≈ 2.0944
            static_cast<uint16_t>((sinf(x + 4.1888f) * 0.5f + 0.5f) * LED_PWM_MAX)  // 4π/3 ≈ 4.1888
        };
        x += speed;
        if(x >= 2 * M_PI) x -= 2 * M_PI;
        led.apply_rgb();

        //if(get_led_status() != LedStatus::RGB)
        //    break;
        if(osMessageQueueGetCount(led_message_queue) != 0)
            break;

        osDelay(10);
    }
}

static void beep_loop(car::Led& led) {
    printf("[led] running LED beep\n");

    led.rgb = {0,0,0};
    while(true) {
        osSemaphoreAcquire(it_timer_sem,osWaitForever);
        led.r = led.r == 0 ? 50 : 0;
        led.apply_rgb();

        //if(get_led_status() != LedStatus::BEEP)
        //    break;
        if(osMessageQueueGetCount(led_message_queue) != 0)
            break;
    }

    HAL_TIM_Base_Stop_IT(&htim3);
}

void led_task(void* args) {
    printf("[led] entrying\n");
    car::Led led(htim4,TIM_CHANNEL_3,TIM_CHANNEL_1,TIM_CHANNEL_2);
    LedStatus status = LedStatus::RGB;
    osMessageQueuePut(led_message_queue,&status,0,0);
    while(true) {
        osMessageQueueGet(led_message_queue,&status,NULL,osWaitForever);
        switch(status) {
        case LedStatus::OFF:
            led.rgb = {0,0,0};
            led.apply_rgb();
            break;
        case LedStatus::KEEP:
            led.rgb = {50,50,50};
            led.apply_rgb();
            break;
        case LedStatus::BEEP:
            beep_loop(led);
            break;
        case LedStatus::RGB:
            rgb_loop(led);
            break;
        }
    }
}