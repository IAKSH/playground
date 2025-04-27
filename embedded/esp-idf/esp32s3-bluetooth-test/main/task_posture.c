#include "task_posture.h"

#include <stdio.h>
#include "driver/i2c.h"

#include "mpu6050.h"
#include "mt9101et.h"
#include "bmp280.h"

#define I2C_MASTER_SCL_IO 1               /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 2               /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0          /*!< I2C port number for master dev */
#define I2C_MASTER_TX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_RX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_FREQ_HZ 100000         /*!< I2C master clock frequency */

Mpu6050Results mpu6050_results;

static void initialize_i2c(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0);
}

void posture_main(void) {
    initialize_i2c();

    mpu6050_init(I2C_MASTER_NUM);
    Mpu6050KalmanState mpu6050_kalman;
    mpu6050_kalman_init(&mpu6050_kalman);

    mt9101et_init();
    Mt9101etKalmanState mt9101et_kalman;
    mt9101et_kalman_init(&mt9101et_kalman);
    uint32_t mt9101et_raw,mt9101et_volt;

    if(!bmp280_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    BMP280KalmanState bmp280_kalman;
    bmp280_kalman_init(&bmp280_kalman);
    float bmp280_press,bmp280_temperature,bmp280_altitude;

    while(true) {
        mpu6050_kalman_update(I2C_MASTER_NUM,&mpu6050_kalman,mpu6050_results.euler,mpu6050_results.accel,&mpu6050_results.temperature);
        mt9101et_kalman_update(&mt9101et_kalman,&mt9101et_raw,&mt9101et_volt);
        bmp280_kalman_update(I2C_MASTER_NUM,&bmp280_kalman,&bmp280_temperature,&bmp280_press,&bmp280_altitude);

        printf("raw=%lu\nvolt=%lu\n",mt9101et_raw,mt9101et_volt);
        mpu6050_results.euler[2] = mt9101et_raw;

        printf("mpu6050's temp = %.2f\n",mpu6050_results.temperature);
        printf("bmp280:\t temp=%.2f\tpress=%.2f\taltitude=%.2f\n",bmp280_temperature,bmp280_press,bmp280_altitude);
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}