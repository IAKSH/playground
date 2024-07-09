#include <stdio.h>
#include <driver/i2c.h>
#include "mpu6050.h"

#define I2C_MASTER_SCL_IO 1               /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 2               /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0          /*!< I2C port number for master dev */
#define I2C_MASTER_TX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_RX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_FREQ_HZ 100000         /*!< I2C master clock frequency */

void initialize_i2c(void) {
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

void app_main(void)
{
    initialize_i2c();
    mpu6050_init(I2C_MASTER_NUM);

    uint8_t id = mpu6050_get_id(I2C_MASTER_NUM);
    if(id != 0x68) {
        printf("can't find mpu6050\n");
        exit(1);
    }
    else {
        printf("found mpu6050, id = %d\n",id);
    }
    
    float euler[3],accel[3],temperature;
    Mpu6050KalmanState mpu6050_kalman;
    mpu6050_kalman_init(&mpu6050_kalman);

    while(true) {
        //mpu6050_get_accel_val(I2C_MASTER_NUM,accel);
        //mpu6050_get_gryo_val(I2C_MASTER_NUM,gryo);
        //mpu6050_get_temperature(I2C_MASTER_NUM,&temperature);
        mpu6050_kalman_update(I2C_MASTER_NUM,&mpu6050_kalman,euler,accel,&temperature);

        printf("accel: x=%.2f\ty=%.2f\tz=%.2f\n",accel[0],accel[1],accel[2]);
        printf("euler: x=%.2f\ty=%.2f\tz=%.2f\n",euler[0],euler[1],euler[2]);
        printf("temperature: %.2f\n",temperature);

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}