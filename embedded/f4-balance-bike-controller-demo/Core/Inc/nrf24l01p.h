#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* User Configurations */
#define NRF24L01P_SPI                     (&hspi1)

#define NRF24L01P_SPI_CS_PIN_PORT         WIRELESS_CSN_GPIO_Port 
#define NRF24L01P_SPI_CS_PIN_NUMBER       WIRELESS_CSN_Pin

#define NRF24L01P_CE_PIN_PORT             WIRELESS_CE_GPIO_Port
#define NRF24L01P_CE_PIN_NUMBER           WIRELESS_CE_Pin

#define NRF24L01P_IRQ_PIN_PORT            EXIT7_WIRELESS_IRQ_GPIO_Port
#define NRF24L01P_IRQ_PIN_NUMBER          EXIT7_WIRELESS_IRQ_Pin

#define NRF24L01P_FRAGMENT_MAX_LEN 128

// 实际上可以改为先发送表示总长度的包，然后发送对应长度的数据
// 至于传一半挂掉的情况，可以在接收端加入超时检测
// 信道利用率似乎更高，但是比目前这个麻烦
typedef struct {
    bool end;
    uint8_t payload[8];
} NRF24L01P_Fragment;

// 1 - 32bytes
#define NRF24L01P_PAYLOAD_LENGTH sizeof(NRF24L01P_Fragment)

typedef enum {
    NRF24L01P_AIR_DATA_RATE_250Kbps = 2,
    NRF24L01P_AIR_DATA_RATE_1Mbps = 0,
    NRF24L01P_AIR_DATA_RATE_2Mbps = 1
} NRF24L01P_AirDataRate;

typedef enum {
    NRF24L01P_OUTPUT_POWER_0dBm = 3,
    NRF24L01P_OUTPUT_POWER_6dBm = 2,
    NRF24L01P_OUTPUT_POWER_12dBm = 1,
    NRF24L01P_OUTPUT_POWER_18dBm = 0,
} NRF24L01P_OutputPower;


void nrf24l01p_set_mode_rx(uint16_t mhz, NRF24L01P_AirDataRate bps);
void nrf24l01p_set_mode_tx(uint16_t mhz, NRF24L01P_AirDataRate bps);

bool nrf24l01p_send_fragment(uint8_t* data);

uint8_t nrf24l01p_write_tx_fifo(uint8_t* tx_payload);
uint8_t nrf24l01p_read_rx_fifo(uint8_t* rx_payload);
void nrf24l01p_clear_rx_dr();

void nrf24l01p_reset();
void nrf24l01p_power_up();
void nrf24l01p_power_down();

uint8_t nrf24l01p_check(void);
void nrf24l01p_set_tx_addr(uint8_t *addr, uint8_t len);
void nrf24l01p_set_rx_addr(uint8_t pipe, uint8_t *addr, uint8_t len);

bool nrf24l01p_check_tx_mode(void);
void nrf24l01p_check_ack(void);

uint8_t nrf24l01p_get_fifo_status(void);

/* nRF24L01+ Commands */
#define NRF24L01P_CMD_R_REGISTER                  0b00000000
#define NRF24L01P_CMD_W_REGISTER                  0b00100000
#define NRF24L01P_CMD_R_RX_PAYLOAD                0b01100001
#define NRF24L01P_CMD_W_TX_PAYLOAD                0b10100000
#define NRF24L01P_CMD_FLUSH_TX                    0b11100001
#define NRF24L01P_CMD_FLUSH_RX                    0b11100010
#define NRF24L01P_CMD_REUSE_TX_PL                 0b11100011
#define NRF24L01P_CMD_R_RX_PL_WID                 0b01100000
#define NRF24L01P_CMD_W_ACK_PAYLOAD               0b10101000
#define NRF24L01P_CMD_W_TX_PAYLOAD_NOACK          0b10110000
#define NRF24L01P_CMD_NOP                         0b11111111    

/* nRF24L01+ Registers */
#define NRF24L01P_REG_CONFIG            0x00
#define NRF24L01P_REG_EN_AA             0x01
#define NRF24L01P_REG_EN_RXADDR         0x02
#define NRF24L01P_REG_SETUP_AW          0x03
#define NRF24L01P_REG_SETUP_RETR        0x04
#define NRF24L01P_REG_RF_CH             0x05
#define NRF24L01P_REG_RF_SETUP          0x06
#define NRF24L01P_REG_STATUS            0x07
#define NRF24L01P_REG_OBSERVE_TX        0x08    // Read-Only
#define NRF24L01P_REG_RPD               0x09    // Read-Only
#define NRF24L01P_REG_RX_ADDR_P0        0x0A
#define NRF24L01P_REG_RX_ADDR_P1        0x0B
#define NRF24L01P_REG_RX_ADDR_P2        0x0C
#define NRF24L01P_REG_RX_ADDR_P3        0x0D
#define NRF24L01P_REG_RX_ADDR_P4        0x0E
#define NRF24L01P_REG_RX_ADDR_P5        0x0F
#define NRF24L01P_REG_TX_ADDR           0x10
#define NRF24L01P_REG_RX_PW_P0          0x11
#define NRF24L01P_REG_RX_PW_P1          0x12
#define NRF24L01P_REG_RX_PW_P2          0x13
#define NRF24L01P_REG_RX_PW_P3          0x14
#define NRF24L01P_REG_RX_PW_P4          0x15
#define NRF24L01P_REG_RX_PW_P5          0x16
#define NRF24L01P_REG_FIFO_STATUS       0x17
#define NRF24L01P_REG_DYNPD             0x1C
#define NRF24L01P_REG_FEATURE           0x1D
