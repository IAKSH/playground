#include "main.h"
#include "driver/nrf24l01p.h"
#include <stdio.h>

#define NRF24L01_TX_DEMO

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
    case EXTI2_MPU_Pin:
        osSemaphoreRelease(gyro_ready_sem);
        __HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_2);
        break;
    case EXIT7_WIRELESS_IRQ_Pin:
#if defined NRF24L01_TX_DEMO
        // 涉及到ACK机制，影响自动重传
        nrf24l01p_tx_irq();
#else if defined  NRF24L01_RX_DEMO
    extern uint8_t rx_data[NRF24L01P_PAYLOAD_LENGTH];
    nrf24l01p_rx_receive(rx_data);
#endif
        __HAL_GPIO_EXTI_CLEAR_IT(EXIT7_WIRELESS_IRQ_Pin);
        break;
    default:
        break;
    }
}