#include <ch32v00x.h>
#include "i2c_interface.h"

static bool i2c_initialized = false;

void i2c_setup(void) {
    if(!i2c_initialized) {
        GPIO_InitTypeDef init_sturcture = {0};
        I2C_InitTypeDef i2c_init_structure = {0};

        RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC | RCC_APB2Periph_AFIO, ENABLE);
        RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1,ENABLE);

        init_sturcture.GPIO_Pin = GPIO_Pin_2;
        init_sturcture.GPIO_Mode = GPIO_Mode_AF_OD;
        init_sturcture.GPIO_Speed = GPIO_Speed_50MHz;
        GPIO_Init(GPIOC,&init_sturcture);

        init_sturcture.GPIO_Pin = GPIO_Pin_1;
        init_sturcture.GPIO_Mode = GPIO_Mode_AF_OD;
        init_sturcture.GPIO_Speed = GPIO_Speed_50MHz;
        GPIO_Init(GPIOC,&init_sturcture);

        i2c_init_structure.I2C_ClockSpeed = 400000;
        i2c_init_structure.I2C_Mode = I2C_Mode_I2C;
        i2c_init_structure.I2C_DutyCycle = I2C_DutyCycle_2;
        i2c_init_structure.I2C_OwnAddress1 = 0x01;
        i2c_init_structure.I2C_Ack = I2C_Ack_Enable;
        i2c_init_structure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
        I2C_Init(I2C1,&i2c_init_structure);

        I2C_Cmd(I2C1,ENABLE);
        I2C_AcknowledgeConfig(I2C1,ENABLE);
    }
    i2c_initialized = true;
}

void i2c_write_reg(uint8_t addr,uint8_t reg_addr,uint8_t* buf,uint16_t len) {
    // Generate a START condition
    I2C_GenerateSTART(I2C1, ENABLE);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

    // Send the device address with a write bit
    I2C_Send7bitAddress(I2C1, addr, I2C_Direction_Transmitter);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));

    // Send the command register address (0x00 for command mode)
    I2C_SendData(I2C1, reg_addr);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

    // Send the actual command byte
    for(uint16_t i = 0;i < len;i++) {
        I2C_SendData(I2C1, buf[i]);
        while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
    }

    // Generate a STOP condition
    I2C_GenerateSTOP(I2C1, ENABLE);
}

void i2c_write(uint8_t addr,uint8_t *buf,uint16_t len) {
    // Generate a START condition
    I2C_GenerateSTART(I2C1, ENABLE);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

    // Send the device address with a write bit
    I2C_Send7bitAddress(I2C1, addr, I2C_Direction_Transmitter);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));

    // Send the actual command byte
    for(uint16_t i = 0;i < len;i++) {
        I2C_SendData(I2C1, buf[i]);
        while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
    }

    // Generate a STOP condition
    I2C_GenerateSTOP(I2C1, ENABLE);
}

void i2c_read(uint8_t addr, uint8_t *buf, uint16_t len) {
    // Generate a START condition
    I2C_GenerateSTART(I2C1, ENABLE);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

    // Send the device address with a read bit
    I2C_Send7bitAddress(I2C1, addr, I2C_Direction_Receiver);
    while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));

    // Read the data
    for (uint16_t i = 0; i < len; i++) {
        // Check if it's the last byte
        if (i == len - 1) {
            // Disable acknowledgment for the last byte
            I2C_AcknowledgeConfig(I2C1, DISABLE);
        }

        while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED));
        buf[i] = I2C_ReceiveData(I2C1);
    }

    // Generate a STOP condition
    I2C_GenerateSTOP(I2C1, ENABLE);

    // Re-enable acknowledgment for future communication
    I2C_AcknowledgeConfig(I2C1, ENABLE);
}
