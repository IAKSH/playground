#include "main.h"
#include <stdio.h>

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
    case GPIO_PIN_2:
        osSemaphoreRelease(gyro_ready_sem);
        __HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_2);
        break;
    default:
        break;
    }
}