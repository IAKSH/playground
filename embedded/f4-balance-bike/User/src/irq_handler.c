#include "main.h"
#include "driver/nrf24l01p.h"
#include "wireless.h"
#include <stdio.h>

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
    case EXTI2_MPU_Pin:
        osSemaphoreRelease(gyro_ready_sem);
        __HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_2);
        break;
    case EXIT7_WIRELESS_IRQ_Pin:
        wireless_irq();
        __HAL_GPIO_EXTI_CLEAR_IT(EXIT7_WIRELESS_IRQ_Pin);
        break;
    default:
        break;
    }
}