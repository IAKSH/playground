#include "driver/nrf24l01p.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define NRF24L01_TX_DEMO

#if defined NRF24L01_TX_DEMO
static uint8_t tx_data[NRF24L01P_PAYLOAD_LENGTH];
static uint8_t tx_address[5] = {0x0,0x0,0x0,0x0,0x01};

typedef struct {
    uint32_t a,b;
} Data;

Data data = {
    .a = 0,
    .b = 0
};

void control_task(void* arg) {
    nrf24l01p_tx_init(2500,_1Mbps);

    if(!nrf24l01p_check()) {
        printf("nrf24l01+ no response\n");
        Error_Handler();
    }

    // 由于需要检查回传的ACK，所以发送端也需要设置rx地址
    nrf24l01p_set_tx_addr(tx_address,5);
    nrf24l01p_set_rx_addr(0,tx_address,5);

    while(1) {
        memcpy(tx_data,&data,NRF24L01P_PAYLOAD_LENGTH);
        data.a++;
        data.b += 2;

        nrf24l01p_tx_transmit(tx_data);

        osDelay(50);
    }
}
#elif defined NRF24L01_RX_DEMO

static uint8_t rx_data[NRF24L01P_PAYLOAD_LENGTH];
static uint8_t rx_address[5] = {0x0,0x0,0x0,0x0,0x01};

typedef struct {
    uint32_t a,b;
} Data;

Data data = {
    .a = 0,
    .b = 0
};

void control_task(void* arg) {
    nrf24l01p_rx_init(2500,_1Mbps);

    if(!nrf24l01p_check()) {
        printf("nrf24l01+ no response\n");
        Error_Handler();
    }

    nrf24l01p_set_rx_addr(0,rx_address,5);

    while(1) {
        printf("rec: {%d,%d}\n",((Data*)rx_data)->a,((Data*)rx_data)->b);
    }
}
#else

void control_task(void* arg) {
    while(1) {

    }
}

#endif