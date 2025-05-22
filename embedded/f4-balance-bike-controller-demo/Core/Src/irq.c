#include "main.h"
#include "wireless.h"

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
    case WIRELESS_IRQ_Pin:
        wireless_irq();
        __HAL_GPIO_EXTI_CLEAR_IT(WIRELESS_IRQ_Pin);
        break;
    default:
        break;
    }
}