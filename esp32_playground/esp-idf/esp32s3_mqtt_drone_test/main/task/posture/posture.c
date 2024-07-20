#include "../tasks.h"

#include "driver/i2c.h"

#include "mpu6050.h"
#include "bmp280.h"
#include "drone_status.h"

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

static Mpu6050KalmanState mpu6050_kalman;
static BMP280KalmanState bmp280_kalman;

static void setup_mpu6050(void) {
    if(!mpu6050_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    mpu6050_kalman_init(&mpu6050_kalman);
}

static void setup_bmp280(void) {
    if(!bmp280_init(I2C_MASTER_NUM)) {
        exit(1);
    }
    bmp280_kalman_init(&bmp280_kalman);
}

void task_posture(void) {
    initialize_i2c();

    setup_mpu6050();
    setup_bmp280();

    while(true) {
        mpu6050_kalman_update(I2C_MASTER_NUM,&mpu6050_kalman,
            drone_gryo_euler,drone_gryo_accel,&drone_gyro_temperature);
        bmp280_kalman_update(I2C_MASTER_NUM,&bmp280_kalman,
            &drone_barometer_pressure,&drone_barometer_altitude,&drone_barometer_temperature);

        printf("euler: %.2f,%.2f,%.2f\n",drone_gryo_euler[0],drone_gryo_euler[1],drone_gryo_euler[2]);

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}