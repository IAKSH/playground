#include "bmp280.h"

#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define BMP280_ADDRESS                  0x76     // BMP280的I2C从机地址

#define BMP280_TEMP_PRESS_CALIB_DATA    0x88  // 温度和气压校准数据
#define BMP280_CHIP_ID                  0xD0  // 芯片ID
#define BMP280_SOFT_RESET               0xE0  // 软复位
#define BMP280_STATUS                   0xF3  // 状态
#define BMP280_CTRL_MEAS                0xF4  // 控制测量
#define BMP280_CONFIG                   0xF5  // 配置
#define BMP280_PRESS_MSB                0xF7  // 气压MSB
#define BMP280_PRESS_LSB                0xF8  // 气压LSB
#define BMP280_PRESS_XLSB               0xF9  // 气压XLSB
#define BMP280_TEMP_MSB                 0xFA  // 温度MSB
#define BMP280_TEMP_LSB                 0xFB  // 温度LSB
#define BMP280_TEMP_XLSB                0xFC  // 温度XLSB

#define BMP280_DIG_T1_LSB_REG 0x88
#define BMP280_DIG_T2_LSB_REG 0x8A
#define BMP280_DIG_T3_LSB_REG 0x8C
#define BMP280_DIG_P1_LSB_REG 0x8E
#define BMP280_DIG_P2_LSB_REG 0x90
#define BMP280_DIG_P3_LSB_REG 0x92
#define BMP280_DIG_P4_LSB_REG 0x94
#define BMP280_DIG_P5_LSB_REG 0x96
#define BMP280_DIG_P6_LSB_REG 0x98
#define BMP280_DIG_P7_LSB_REG 0x9A
#define BMP280_DIG_P8_LSB_REG 0x9C
#define BMP280_DIG_P9_LSB_REG 0x9E

typedef struct {
    uint16_t dig_t1;
    int16_t dig_t2;
    int16_t dig_t3;
    uint16_t dig_p1;
    int16_t dig_p2;
    int16_t dig_p3;
    int16_t dig_p4;
    int16_t dig_p5;
    int16_t dig_p6;
    int16_t dig_p7;
    int16_t dig_p8;
    int16_t dig_p9;
} bmp280_calib_param_t;

static bmp280_calib_param_t bmp280_calib_param;

static esp_err_t bmp280_read(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t *data, size_t data_len) {
    if (data_len == 0) {
        return ESP_OK;
    }
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (BMP280_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (BMP280_ADDRESS << 1) | I2C_MASTER_READ, true);
    if (data_len > 1) {
        i2c_master_read(cmd, data, data_len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + data_len - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t bmp280_write(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (BMP280_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static void bmp280_get_calib_param(i2c_port_t i2c_num, bmp280_calib_param_t* param) {
    uint8_t buffer[2];

    // 读取温度校准参数
    bmp280_read(i2c_num, BMP280_DIG_T1_LSB_REG, buffer, 2);
    param->dig_t1 = (uint16_t)buffer[0] | ((uint16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_T2_LSB_REG, buffer, 2);
    param->dig_t2 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_T3_LSB_REG, buffer, 2);
    param->dig_t3 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    // 读取气压校准参数
    bmp280_read(i2c_num, BMP280_DIG_P1_LSB_REG, buffer, 2);
    param->dig_p1 = (uint16_t)buffer[0] | ((uint16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P2_LSB_REG, buffer, 2);
    param->dig_p2 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P3_LSB_REG, buffer, 2);
    param->dig_p3 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P4_LSB_REG, buffer, 2);
    param->dig_p4 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P5_LSB_REG, buffer, 2);
    param->dig_p5 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P6_LSB_REG, buffer, 2);
    param->dig_p6 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P7_LSB_REG, buffer, 2);
    param->dig_p7 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P8_LSB_REG, buffer, 2);
    param->dig_p8 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    bmp280_read(i2c_num, BMP280_DIG_P9_LSB_REG, buffer, 2);
    param->dig_p9 = (int16_t)buffer[0] | ((int16_t)buffer[1] << 8);

    printf("BMP280 param->dig_T1 = %u\n",param->dig_t1);
    printf("BMP280 param->dig_T2 = %d\n",param->dig_t2);
    printf("BMP280 param->dig_T3 = %d\n",param->dig_t3);
    printf("BMP280 param->dig_P1 = %u\n",param->dig_p1);
    printf("BMP280 param->dig_P2 = %d\n",param->dig_p2);
    printf("BMP280 param->dig_P3 = %d\n",param->dig_p3);
    printf("BMP280 param->dig_P4 = %d\n",param->dig_p4);
    printf("BMP280 param->dig_P5 = %d\n",param->dig_p5);
    printf("BMP280 param->dig_P6 = %d\n",param->dig_p6);
    printf("BMP280 param->dig_P7 = %d\n",param->dig_p7);
    printf("BMP280 param->dig_P8 = %d\n",param->dig_p8);
    printf("BMP280 param->dig_P9 = %d\n",param->dig_p9);
}

static uint8_t bmp280_get_id(i2c_port_t i2c_num) {
    uint8_t id;
    bmp280_read(i2c_num, BMP280_CHIP_ID, &id, sizeof(uint8_t));
    return id;
}

bool bmp280_init(i2c_port_t i2c_num) {

    bmp280_write(i2c_num, BMP280_SOFT_RESET, 0xB6);  // 软复位
    vTaskDelay(pdMS_TO_TICKS(200));  // 等待复位完成

    int id = bmp280_get_id(i2c_num);
    if(id == 0x58) {
        printf("found bmp280\n");
    }
    else {
        printf("bmp280 not found, using address = 0x%x, got id = 0x%x\n",BMP280_ADDRESS,id);
        return false;
    }
    
    bmp280_write(i2c_num, BMP280_CTRL_MEAS, 0xFF);  // 设置控制测量寄存器，使能温度和气压测量
    bmp280_write(i2c_num, BMP280_CONFIG, 0x00);  // 设置配置寄存器，使用默认设置

    bmp280_get_calib_param(i2c_num,&bmp280_calib_param);
    vTaskDelay(pdMS_TO_TICKS(200));

    return true;
}

void bmp280_get_raw_temp_press(i2c_port_t i2c_num, int32_t *temp, int32_t *press) {
    uint8_t data[6];
    bmp280_read(i2c_num, BMP280_PRESS_MSB, data, 6);
    *press = ((int32_t)data[0] << 12) | ((int32_t)data[1] << 4) | ((int32_t)data[2] >> 4);
    *temp = ((int32_t)data[3] << 12) | ((int32_t)data[4] << 4) | ((int32_t)data[5] >> 4);
}

// 单位是百分之一摄氏度和256Pa
void bmp280_get_temp_press(i2c_port_t i2c_num, int32_t *temp, uint32_t *press) {
    int32_t t_fine, press_raw, temp_raw;
    bmp280_get_raw_temp_press(i2c_num,&temp_raw,&press_raw);

    int32_t var1_t, var2_t, t;
    int64_t var1_p, var2_p, p;
    var1_t = ((((temp_raw>>3) - ((int32_t)bmp280_calib_param.dig_t1 <<1))) * ((int32_t)bmp280_calib_param.dig_t2)) >> 11;
    var2_t = (((((temp_raw>>4) - ((int32_t)bmp280_calib_param.dig_t1)) * ((temp_raw>>4) - ((int32_t)bmp280_calib_param.dig_t1))) >> 12) * ((int32_t)bmp280_calib_param.dig_t3)) >> 14;
    t_fine = var1_t + var2_t;
    t = (t_fine * 5 + 128) >> 8;
    *temp = t;

    var1_p = ((int64_t)t_fine) - 128000;
    var2_p = var1_p * var1_p * (int64_t)bmp280_calib_param.dig_p6;
    var2_p = var2_p + ((var1_p*(int64_t)bmp280_calib_param.dig_p5)<<17);
    var2_p = var2_p + (((int64_t)bmp280_calib_param.dig_p4)<<35);
    var1_p = ((var1_p * var1_p * (int64_t)bmp280_calib_param.dig_p3)>>8) + ((var1_p * (int64_t)bmp280_calib_param.dig_p2)<<12);
    var1_p = (((((int64_t)1)<<47)+var1_p))*((int64_t)bmp280_calib_param.dig_p1)>>33;

    p = 1048576 - press_raw;
    p = (((p<<31)-var2_p)*3125)/var1_p;
    var1_p = (((int64_t)bmp280_calib_param.dig_p9) * (p>>13) * (p>>13)) >> 25;
    var2_p = (((int64_t)bmp280_calib_param.dig_p8) * p) >> 19;
    p = ((p + var1_p + var2_p) >> 8) + (((int64_t)bmp280_calib_param.dig_p7)<<4);
    *press = (uint32_t)p;
}

#define SEA_LEVEL_PRESSURE 1013.25

void bmp280_press_temp_to_altitude(uint32_t press, int32_t temperature, float* altitude) {
    float press_in_hpa = press / 25600.0;
    float temperature_in_celsius = temperature / 100.0;
    *altitude = 44330.0 * (1.0 - pow(press_in_hpa / SEA_LEVEL_PRESSURE, 1/5.255));
}

void bmp280_get_altitude(i2c_port_t i2c_num, float *altitude) {
    int32_t temp;
    uint32_t press;
    bmp280_get_temp_press(i2c_num, &temp, &press);  // 读取温度和气压数据
    bmp280_press_temp_to_altitude(press,temp,altitude);
}

void bmp280_kalman_init(bmp280_kalman_state_t* kalman_state) {
    kalman_init(&kalman_state->altitude, 0.1, 0.1, 0.1, 0);
    kalman_init(&kalman_state->press, 0.1, 0.1, 0.1, 0);
    kalman_init(&kalman_state->temperature, 0.1, 0.1, 0.1, 0);
}

void bmp280_kalman_update(i2c_port_t i2c_num,bmp280_kalman_state_t* kalman_state,float* press,float* altitude,float* temperature) {
    int32_t mesure_temperature;
    uint32_t mesure_press;
    float mesure_altitude;
    bmp280_get_temp_press(i2c_num,&mesure_temperature,&mesure_press);
    bmp280_press_temp_to_altitude(mesure_press,mesure_temperature,&mesure_altitude);

    *altitude = kalman_update(&kalman_state->altitude,mesure_altitude);
    *press = kalman_update(&kalman_state->press,mesure_press / 25600.0);
    *temperature = kalman_update(&kalman_state->temperature,mesure_temperature / 100.0);
}