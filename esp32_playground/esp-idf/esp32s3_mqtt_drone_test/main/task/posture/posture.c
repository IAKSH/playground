#include "../tasks.h"

#include "driver/i2c.h"
#include "esp_timer.h"

#include "mpu6050.h"
#include "bmp280.h"
#include "drone_status.h"

#include <math.h>

#define I2C_MASTER_SCL_IO 1               /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 2               /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0          /*!< I2C port number for master dev */
#define I2C_MASTER_TX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_RX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_FREQ_HZ 100000         /*!< I2C master clock frequency */

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

void task_posture(void) {
    initialize_i2c();

    mpu6050_kalman_state_t mpu6050_kalman_state[2];
    //BMP280KalmanState bmp280_kalman;

    if(!mpu6050_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    mpu6050_kalman_init(&mpu6050_kalman_state,0,0,0);
    
    //if(!bmp280_init(I2C_MASTER_NUM)) {
    //    exit(1);
    //}
    //bmp280_kalman_init(&bmp280_kalman);

    while(true) {
        static int64_t last_time = 0;
        int64_t current_time = esp_timer_get_time();
        float dt = (current_time - last_time) / 1000000.0;
        last_time = current_time;

        float accel[3],gyro[3];
        mpu6050_get_accel_val(I2C_MASTER_NUM,accel);
        mpu6050_get_gyro_val(I2C_MASTER_NUM,gyro);

        float accel_angle[2] = {atan2(accel[1], accel[2]) * 180 / M_PI,atan2(-accel[0], sqrt(accel[1] * accel[1] + accel[2] * accel[2])) * 180 / M_PI};
        
        for(int i = 0;i < 2;i++) {
            mpu6050_kalman_update(&mpu6050_kalman_state[i],gyro[i],accel_angle[i],dt);
            drone_gryo_euler[i] = mpu6050_kalman_state[i].angle;
        }

        //drone_gryo_euler[0] = atan2(-drone_gryo_accel[0], sqrt(drone_gryo_accel[1] * drone_gryo_accel[1] + drone_gryo_accel[2] * drone_gryo_accel[2])) * 180 / M_PI;
        //drone_gryo_euler[1] = atan2(drone_gryo_accel[1], drone_gryo_accel[2]) * 180 / M_PI;

        //bmp280_kalman_update(I2C_MASTER_NUM,&bmp280_kalman,                                                                         
        //    &drone_barometer_pressure,&drone_barometer_altitude,&drone_barometer_temperature);

        //printf("euler: %.2f,%.2f,%.2f\n",drone_gryo_euler[0],drone_gryo_euler[1],drone_gryo_euler[2]);
    }
}