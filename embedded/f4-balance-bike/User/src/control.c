#include "driver/nrf24l01p.h"
#include <stdio.h>
#include <stdint.h>

static uint8_t tx_data[NRF24L01P_PAYLOAD_LENGTH] = {0,1,2,3,4,5,6,7};
uint8_t rx_data[NRF24L01P_PAYLOAD_LENGTH] = {0};

void control_task(void* arg) {
    nrf24l01p_tx_init(2500,_1Mbps);
    nrf24l01p_rx_init(2500,_1Mbps);

    printf("nrf24l01p intialized\n");
    printf("running nrf24l01p test\n");

    while(1) {
        for(int i = 0;i < 8;i++)
            tx_data[i]++;

        nrf24l01p_tx_transmit(tx_data);
        osDelay(100);
    }
}