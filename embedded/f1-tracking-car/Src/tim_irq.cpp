#include "tim_irq.h"
#include "main.h"
#include "status.hpp"

void tim_irq(TIM_HandleTypeDef *htim) {
    if(htim->Instance == TIM3) {
        osSemaphoreRelease(it_timer_sem);
        osSemaphoreRelease(sensor_timer_sem);
    }
    if(htim->Instance == TIM5) {
        ticker.irq();
    }
}