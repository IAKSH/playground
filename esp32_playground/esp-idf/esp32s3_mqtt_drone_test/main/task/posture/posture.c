#include "../tasks.h"

#include "driver/i2c.h"
#include "esp_timer.h"

#include "mpu6050.h"
#include "bmp280.h"
#include "qmc5883l.h"
#include "drone_status.h"

#include <math.h>

#define I2C_MASTER_SCL_IO 1               /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 2               /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0          /*!< I2C port number for master dev */
#define I2C_MASTER_TX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_RX_BUF_DISABLE 0       /*!< I2C master do not need buffer */
#define I2C_MASTER_FREQ_HZ 100000         /*!< I2C master clock frequency */

#define ENABLE_MPU6050

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

# ifdef ENABLE_MPU6050
    mpu6050_kalman_state_t mpu6050_kalman_state[2];
#endif
    bmp280_kalman_state_t bmp280_kalman_state;

# ifdef ENABLE_MPU6050
    if(!mpu6050_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    for(int i = 0;i < 2;i++)
        mpu6050_kalman_init(mpu6050_kalman_state + i,0,0,0);
#endif

    if(!bmp280_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    bmp280_kalman_init(&bmp280_kalman_state);

    if(!qmc5883l_init(I2C_MASTER_NUM)) {
        exit(1);
    }

    while(true) {
        static int64_t last_time = 0;
        int64_t current_time = esp_timer_get_time();
        float dt = (current_time - last_time) / 1000000.0;
        last_time = current_time;

# ifdef ENABLE_MPU6050
        float accel[3],gyro[3];
        mpu6050_get_accel_val(I2C_MASTER_NUM,accel);
        mpu6050_get_gyro_val(I2C_MASTER_NUM,gyro);
        float accel_angle[2] = {atan2(accel[1], accel[2]) * 180 / M_PI,atan2(-accel[0], sqrt(accel[1] * accel[1] + accel[2] * accel[2])) * 180 / M_PI};
        for(int i = 0;i < 2;i++) {
            mpu6050_kalman_update(&mpu6050_kalman_state[i],gyro[i],accel_angle[i],dt);
        }
        drone_gryo_euler[0] = -mpu6050_kalman_state[1].angle;
        drone_gryo_euler[1] = mpu6050_kalman_state[0].angle;
        //drone_gryo_euler[2] = qmc5883l_get_filted_yaw(I2C_MASTER_NUM);
        mpu6050_get_temperature(I2C_MASTER_NACK,&drone_gyro_temperature);
        drone_gryo_euler[2] = qmc5883l_get_yaw(I2C_MASTER_NUM);
#endif

        bmp280_kalman_update(I2C_MASTER_NUM,&bmp280_kalman_state,                                                                         
            &drone_barometer_pressure,&drone_barometer_altitude,&drone_barometer_temperature);

        //printf("pressure: %.2f\taltitude: %.2f\ttemperature: %.2f\n",drone_barometer_pressure,drone_barometer_altitude,drone_barometer_temperature);
    }
}