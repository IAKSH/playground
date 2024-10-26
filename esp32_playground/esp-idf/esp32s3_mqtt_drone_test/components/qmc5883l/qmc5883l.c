#include "qmc5883l.h"

#include <math.h>
#include <stdio.h>

#define QMC5883_REG_DATA       0x00
#define QMC5883_REG_OUT_X_L    0x00
#define QMC5883_REG_OUT_X_M    0x01
#define QMC5883_REG_OUT_Y_L    0x02
#define QMC5883_REG_OUT_Y_M    0x03
#define QMC5883_REG_OUT_Z_L    0x04
#define QMC5883_REG_OUT_Z_M    0x05
 
#define QMC5883_REG_STATUS     0x06
#define QMC5883_DRDY_BIT0      //0: no new data, 1: new data is ready
#define QMC5883_OVL_BIT1       //0: normal,      1: data overflow
#define QMC5883_DOR_BIT2       //0: normal,      1: data skipped for reading
   
#define QMC5883_REG_TEMP_OUT_L 0x07
#define QMC5883_REG_TEMP_OUT_H 0x08
 
#define QMC5883_REG_CTRL1      0x09
#define QMC5883_CMD_MODE_STANDBY     0x00  //mode 
#define QMC5883_CMD_MODE_CON         0x01
#define QMC5883_CMD_ODR_10HZ         0x00  //Output Data Rate
#define QMC5883_CMD_ODR_50HZ         0x04
#define QMC5883_CMD_ODR_100HZ        0x08
#define QMC5883_CMD_ODR_200HZ        0x0C
#define QMC5883_CMD_RNG_2G           0x00  //Full Scale
#define QMC5883_CMD_RNG_8G           0x10    
#define QMC5883_CMD_OSR_512          0x00  //Over Sample Ratio
#define QMC5883_CMD_OSR_256          0x40    
#define QMC5883_CMD_OSR_128          0x80    
#define QMC5883_CMD_OSR_64           0xC0    
 
#define QMC5883_REG_CTRL2      0x0A
#define QMC5883_CMD_INT_ENABLE       0x00 
#define QMC5883_CMD_INT_DISABLE      0x01
#define QMC5883_CMD_ROL_PNT_ENABLE   0x40  //pointer roll-over function,only 0x00-0x06 address
#define QMC5883_CMD_INT_ENABLE       0x00 
#define QMC5883_CMD_INT_DISABLE      0x01
#define QMC5883_CMD_ROL_PNT_ENABLE   0x40  //pointer roll-over function,only 0x00-0x06 address
#define QMC5883_CMD_ROL_PNT_DISABLE  0x00 
#define QMC5883_CMD_SOFT_RST_ENABLE  0x80
#define QMC5883_CMD_SOFT_RST_DISABLE 0x00 
   
#define QMC5883_REG_SET_RESET  0x0B
#define QMC5883_CMD_SET_RESET        0x01 
  
#define QMC5883_REG_PRODUCTID  0x0D           //chip id :0xFF

#define QMC5883L_ADDRESS QMC5883_REG_PRODUCTID

#define BYTE_TO_HALFWORD(high,low) (int16_t)(high << 8 | low)

static esp_err_t qmc5883l_write(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (QMC5883L_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

// 读取QMC5883L寄存器
static esp_err_t qmc5883l_read(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t *data, size_t data_len) {
    if (data_len == 0) {
        return ESP_OK;
    }
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (QMC5883L_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (QMC5883L_ADDRESS << 1) | I2C_MASTER_READ, true);
    if (data_len > 1) {
        i2c_master_read(cmd, data, data_len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + data_len - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static uint8_t qmc5883l_get_device_id(i2c_port_t i2c_num) {
    uint8_t device_id;
    qmc5883l_read(i2c_num, QMC5883L_ADDRESS, &device_id, sizeof(uint8_t));
    return device_id;
}

bool qmc5883l_init(i2c_port_t i2c_num) {
    if(qmc5883l_get_device_id(i2c_num) != 0xff) {
        printf("can't find qmc5883l!\n");
        return false;
    }

    qmc5883l_write(i2c_num, QMC5883_REG_CTRL2, QMC5883_CMD_SOFT_RST_ENABLE);
    qmc5883l_write(i2c_num, QMC5883_REG_CTRL1, QMC5883_CMD_MODE_CON | QMC5883_CMD_ODR_10HZ | QMC5883_CMD_RNG_8G | QMC5883_CMD_OSR_512);
    qmc5883l_write(i2c_num, QMC5883_REG_CTRL2, QMC5883_CMD_INT_DISABLE | QMC5883_CMD_ROL_PNT_ENABLE);

    return true;
}

float qmc5883l_get_yaw(i2c_port_t i2c_num) {
    return 0;
}