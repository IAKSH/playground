#include "main.h"
#include "nrf24l01p.h"

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
    case WIRELESS_IRQ_Pin:
        // 涉及到ACK机制，影响自动重传
        nrf24l01p_tx_irq();
        __HAL_GPIO_EXTI_CLEAR_IT(WIRELESS_IRQ_Pin);
        break;
    default:
        break;
    }
}