#include "nrf24l01p.h"

static uint8_t nrf24l01p_tx_result = 0;

void nrf24l01p_clear_tx_result(void) {
    nrf24l01p_tx_result = 0;
}

uint8_t nrf24l01p_get_tx_result(void) {
    return nrf24l01p_tx_result;
}

static void cs_high()
{
    HAL_GPIO_WritePin(NRF24L01P_SPI_CS_PIN_PORT, NRF24L01P_SPI_CS_PIN_NUMBER, GPIO_PIN_SET);
}

static void cs_low()
{
    HAL_GPIO_WritePin(NRF24L01P_SPI_CS_PIN_PORT, NRF24L01P_SPI_CS_PIN_NUMBER, GPIO_PIN_RESET);
}

static void ce_high()
{
    HAL_GPIO_WritePin(NRF24L01P_CE_PIN_PORT, NRF24L01P_CE_PIN_NUMBER, GPIO_PIN_SET);
}

static void ce_low()
{
    HAL_GPIO_WritePin(NRF24L01P_CE_PIN_PORT, NRF24L01P_CE_PIN_NUMBER, GPIO_PIN_RESET);
}

static uint8_t read_register(uint8_t reg)
{
    uint8_t command = NRF24L01P_CMD_R_REGISTER | reg;
    uint8_t status;
    uint8_t read_val;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    HAL_SPI_Receive(NRF24L01P_SPI, &read_val, 1, 2000);
    cs_high();

    return read_val;
}

static uint8_t write_register(uint8_t reg, uint8_t value)
{
    uint8_t command = NRF24L01P_CMD_W_REGISTER | reg;
    uint8_t status;
    uint8_t write_val = value;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    HAL_SPI_Transmit(NRF24L01P_SPI, &write_val, 1, 2000);
    cs_high();

    return write_val;
}

static void write_register_multi(uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t command = NRF24L01P_CMD_W_REGISTER | reg;

    cs_low();
    HAL_SPI_Transmit(NRF24L01P_SPI, &command, 1, 2000);
    HAL_SPI_Transmit(NRF24L01P_SPI, buf, len, 2000);
    cs_high();
}

static void read_register_multi(uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t command = NRF24L01P_CMD_R_REGISTER | reg;

    cs_low();
    HAL_SPI_Transmit(NRF24L01P_SPI, &command, 1, 2000);
    HAL_SPI_Receive(NRF24L01P_SPI, buf, len, 2000);
    cs_high();
}

void nrf24l01p_rx_init(channel MHz, air_data_rate bps)
{
    nrf24l01p_reset();

    nrf24l01p_prx_mode();
    nrf24l01p_power_up();

    nrf24l01p_rx_set_payload_widths(NRF24L01P_PAYLOAD_LENGTH);

    nrf24l01p_set_rf_channel(MHz);
    nrf24l01p_set_rf_air_data_rate(bps);
    nrf24l01p_set_rf_tx_output_power(_0dBm);

    nrf24l01p_set_crc_length(1);
    nrf24l01p_set_address_widths(5);

    nrf24l01p_auto_retransmit_count(3);
    nrf24l01p_auto_retransmit_delay(250);
    
    ce_high();
}

void nrf24l01p_tx_init(channel MHz, air_data_rate bps)
{
    nrf24l01p_reset();

    nrf24l01p_ptx_mode();
    nrf24l01p_power_up();

    nrf24l01p_set_rf_channel(MHz);
    nrf24l01p_set_rf_air_data_rate(bps);
    nrf24l01p_set_rf_tx_output_power(_0dBm);

    nrf24l01p_set_crc_length(1);
    nrf24l01p_set_address_widths(5);

    nrf24l01p_auto_retransmit_count(3);
    nrf24l01p_auto_retransmit_delay(250);

    ce_high();
}

uint8_t nrf24l01p_check(void)
{
    uint8_t testValue[5] = {0xA5, 0xA5, 0xA5, 0xA5, 0xA5};
    uint8_t readValue[5] = {0};
    
    // 写入 TX 地址寄存器测试数据
    write_register_multi(NRF24L01P_REG_TX_ADDR, testValue, 5);
    
    // 读回 TX 地址寄存器数据
    read_register_multi(NRF24L01P_REG_TX_ADDR, readValue, 5);
    
    // 对比数据是否一致
    for (int i = 0; i < 5; i++)
    {
        if (readValue[i] != testValue[i])
        {
            return 0;   // 检测失败，返回 0 表示模块可能未连接或响应异常
        }
    }
    return 1;   // 数据匹配，模块正常响应
}

/*
 * 设置发送端 TX 地址
 *
 * 参数：
 *   addr：地址数据指针（通常为 5 字节地址）
 *   len : 地址长度（一般为 5）
 *
 * 使用说明：
 *   调用此函数设置 nRF24L01+ 的 TX_ADDR 寄存器，确保发送端的地址
 *   与接收端接收通道的地址一致。
 */
void nrf24l01p_set_tx_addr(uint8_t *addr, uint8_t len)
{
    /* 可在必要时添加对 len 有效性判断，比如限制为5 */
    write_register_multi(NRF24L01P_REG_TX_ADDR, addr, len);
}

/*
 * 设置接收端对应的数据通道接收地址
 *
 * 参数：
 *   pipe：指定使用的接收通道 (0 ~ 5)
 *   addr：地址数据指针
 *   len : 地址长度
 *
 * 使用说明：
 *   对于pipe0和pipe1，nRF24L01+支持完整地址（通常为5字节）；而对于pipe2到pipe5，
 *   硬件只允许配置第一个字节（其他4字节与pipe1重复），因此这里建议：
 *     - 对于 pipe0 和 pipe1，len 应为 5
 *     - 对于 pipe2~pipe5，len 应为 1
 */
void nrf24l01p_set_rx_addr(uint8_t pipe, uint8_t *addr, uint8_t len)
{
    uint8_t reg;
    
    if (pipe == 0)
    {
        reg = NRF24L01P_REG_RX_ADDR_P0;
    }
    else if (pipe == 1)
    {
        reg = NRF24L01P_REG_RX_ADDR_P1;
    }
    else if (pipe >= 2 && pipe <= 5)
    {
        /* 对于 pipe2~pipe5，只有最低有效字节可配置，其他字节与 pipe1 相同 */
        if (len != 1)
        {
            /* 非法长度，只写入1字节 */
            return;
        }
        reg = NRF24L01P_REG_RX_ADDR_P0 + pipe;
    }
    else
    {
        /* 超出有效通道范围 */
        return;
    }
    
    write_register_multi(reg, addr, len);
}

void nrf24l01p_rx_receive(uint8_t* rx_payload)
{
    nrf24l01p_read_rx_fifo(rx_payload);
    nrf24l01p_clear_rx_dr();

    //HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
}

void nrf24l01p_tx_transmit(uint8_t* tx_payload)
{
    nrf24l01p_write_tx_fifo(tx_payload);
    //ce_high();
    //HAL_Delay(1); // 或适当延时>10us
    //ce_low();
}

void nrf24l01p_tx_irq()
{
    uint8_t status = nrf24l01p_get_status();

    if(status & 0x20) // TX_DS
    {   
        nrf24l01p_tx_result = 1;
        nrf24l01p_clear_tx_ds();
    }
    else if(status & 0x10) // MAX_RT
    {
        nrf24l01p_tx_result = 0;
        nrf24l01p_clear_max_rt();
    }
}

/* nRF24L01+ Sub Functions */
void nrf24l01p_reset()
{
    // Reset pins
    cs_high();
    ce_low();

    // Reset registers
    write_register(NRF24L01P_REG_CONFIG, 0x08);
    write_register(NRF24L01P_REG_EN_AA, 0x3F);
    write_register(NRF24L01P_REG_EN_RXADDR, 0x03);
    write_register(NRF24L01P_REG_SETUP_AW, 0x03);
    write_register(NRF24L01P_REG_SETUP_RETR, 0x03);
    write_register(NRF24L01P_REG_RF_CH, 0x02);
    write_register(NRF24L01P_REG_RF_SETUP, 0x07);
    write_register(NRF24L01P_REG_STATUS, 0x7E);
    write_register(NRF24L01P_REG_RX_PW_P0, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P0, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P1, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P2, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P3, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P4, 0x00);
    write_register(NRF24L01P_REG_RX_PW_P5, 0x00);
    write_register(NRF24L01P_REG_FIFO_STATUS, 0x11);
    write_register(NRF24L01P_REG_DYNPD, 0x00);
    write_register(NRF24L01P_REG_FEATURE, 0x00);

    // Reset FIFO
    nrf24l01p_flush_rx_fifo();
    nrf24l01p_flush_tx_fifo();
}

void nrf24l01p_prx_mode()
{
    uint8_t new_config = read_register(NRF24L01P_REG_CONFIG);
    new_config |= 1 << 0;

    write_register(NRF24L01P_REG_CONFIG, new_config);
}

void nrf24l01p_ptx_mode()
{
    uint8_t new_config = read_register(NRF24L01P_REG_CONFIG);
    new_config &= 0xFE;

    write_register(NRF24L01P_REG_CONFIG, new_config);
}

uint8_t nrf24l01p_read_rx_fifo(uint8_t* rx_payload)
{
    uint8_t command = NRF24L01P_CMD_R_RX_PAYLOAD;
    uint8_t status;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    HAL_SPI_Receive(NRF24L01P_SPI, rx_payload, NRF24L01P_PAYLOAD_LENGTH, 2000);
    cs_high();

    return status;
}

uint8_t nrf24l01p_write_tx_fifo(uint8_t* tx_payload)
{
    uint8_t command = NRF24L01P_CMD_W_TX_PAYLOAD;
    uint8_t status;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    HAL_SPI_Transmit(NRF24L01P_SPI, tx_payload, NRF24L01P_PAYLOAD_LENGTH, 2000);
    cs_high(); 

    return status;
}

void nrf24l01p_flush_rx_fifo()
{
    uint8_t command = NRF24L01P_CMD_FLUSH_RX;
    uint8_t status;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    cs_high();
}

void nrf24l01p_flush_tx_fifo()
{
    uint8_t command = NRF24L01P_CMD_FLUSH_TX;
    uint8_t status;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    cs_high();
}

uint8_t nrf24l01p_get_status()
{
    uint8_t command = NRF24L01P_CMD_NOP;
    uint8_t status;

    cs_low();
    HAL_SPI_TransmitReceive(NRF24L01P_SPI, &command, &status, 1, 2000);
    cs_high(); 

    return status;
}

uint8_t nrf24l01p_get_fifo_status()
{
    return read_register(NRF24L01P_REG_FIFO_STATUS);
}

void nrf24l01p_rx_set_payload_widths(widths bytes)
{
    write_register(NRF24L01P_REG_RX_PW_P0, bytes);
}

void nrf24l01p_clear_rx_dr()
{
    uint8_t new_status = nrf24l01p_get_status();
    new_status |= 0x40;

    write_register(NRF24L01P_REG_STATUS, new_status);
}

void nrf24l01p_clear_tx_ds()
{
    uint8_t new_status = nrf24l01p_get_status();
    new_status |= 0x20;

    write_register(NRF24L01P_REG_STATUS, new_status);     
}

void nrf24l01p_clear_max_rt()
{
    uint8_t new_status = nrf24l01p_get_status();
    new_status |= 0x10;

    write_register(NRF24L01P_REG_STATUS, new_status); 
}

void nrf24l01p_power_up()
{
    uint8_t new_config = read_register(NRF24L01P_REG_CONFIG);
    new_config |= 1 << 1;

    write_register(NRF24L01P_REG_CONFIG, new_config);
}

void nrf24l01p_power_down()
{
    uint8_t new_config = read_register(NRF24L01P_REG_CONFIG);
    new_config &= 0xFD;

    write_register(NRF24L01P_REG_CONFIG, new_config);
}

void nrf24l01p_set_crc_length(length bytes)
{
    uint8_t new_config = read_register(NRF24L01P_REG_CONFIG);
    
    switch(bytes)
    {
        // CRCO bit in CONFIG resiger set 0
        case 1:
            new_config &= 0xFB;
            break;
        // CRCO bit in CONFIG resiger set 1
        case 2:
            new_config |= 1 << 2;
            break;
    }

    write_register(NRF24L01P_REG_CONFIG, new_config);
}

void nrf24l01p_set_address_widths(widths bytes)
{
    write_register(NRF24L01P_REG_SETUP_AW, bytes - 2);
}

void nrf24l01p_auto_retransmit_count(count cnt)
{
    uint8_t new_setup_retr = read_register(NRF24L01P_REG_SETUP_RETR);
    
    // Reset ARC register 0
    new_setup_retr |= 0xF0;
    new_setup_retr |= cnt;
    write_register(NRF24L01P_REG_SETUP_RETR, new_setup_retr);
}

void nrf24l01p_auto_retransmit_delay(delay us)
{
    uint8_t new_setup_retr = read_register(NRF24L01P_REG_SETUP_RETR);

    // Reset ARD register 0
    new_setup_retr |= 0x0F;
    new_setup_retr |= ((us / 250) - 1) << 4;
    write_register(NRF24L01P_REG_SETUP_RETR, new_setup_retr);
}

void nrf24l01p_set_rf_channel(channel MHz)
{
    uint16_t new_rf_ch = MHz - 2400;
    write_register(NRF24L01P_REG_RF_CH, new_rf_ch);
}

void nrf24l01p_set_rf_tx_output_power(output_power dBm)
{
    uint8_t new_rf_setup = read_register(NRF24L01P_REG_RF_SETUP) & 0xF9;
    new_rf_setup |= (dBm << 1);

    write_register(NRF24L01P_REG_RF_SETUP, new_rf_setup);
}

void nrf24l01p_set_rf_air_data_rate(air_data_rate bps)
{
    // Set value to 0
    uint8_t new_rf_setup = read_register(NRF24L01P_REG_RF_SETUP) & 0xD7;
    
    switch(bps)
    {
        case _1Mbps: 
            break;
        case _2Mbps: 
            new_rf_setup |= 1 << 3;
            break;
        case _250kbps:
            new_rf_setup |= 1 << 5;
            break;
    }
    write_register(NRF24L01P_REG_RF_SETUP, new_rf_setup);
}