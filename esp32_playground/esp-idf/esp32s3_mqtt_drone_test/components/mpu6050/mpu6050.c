#include "mpu6050.h"
#include <math.h>

#define MPU6050_ADDRESS             0x69     //MPU6050的I2C从机地址
 
#define MPU6050_SMPLRT_DIV              0x19     //采样频率分频器
#define MPU6050_CONFIG                  0x1A     //配置
#define MPU6050_GYRO_CONFIG             0x1B     //陀螺仪配置
#define MPU6050_ACCEL_CONFIG    0x1C     //加速度计配置
 
#define MPU6050_ACCEL_XOUT_H    0x3B     //加速度计X轴高位
#define MPU6050_ACCEL_XOUT_L    0x3C     //加速度计X轴底位
#define MPU6050_ACCEL_YOUT_H    0x3D     //加速度计Y轴高位
#define MPU6050_ACCEL_YOUT_L    0x3E     //加速度计Y轴底位
#define MPU6050_ACCEL_ZOUT_H    0x3F     //加速度计Z轴高位
#define MPU6050_ACCEL_ZOUT_L    0x40     //加速度计Z轴底位
#define MPU6050_TEMP_OUT_H              0x41     //温度测量值高位     
#define MPU6050_TEMP_OUT_L              0x42     //温度测量值低位
#define MPU6050_GYRO_XOUT_H             0x43     //陀螺仪X轴高位
#define MPU6050_GYRO_XOUT_L             0x44     //陀螺仪X轴低位
#define MPU6050_GYRO_YOUT_H             0x45     //陀螺仪Y轴高位
#define MPU6050_GYRO_YOUT_L             0x46     //陀螺仪Y轴低位
#define MPU6050_GYRO_ZOUT_H             0x47     //陀螺仪Z轴高位
#define MPU6050_GYRO_ZOUT_L             0x48     //陀螺仪Z轴低位
 
#define MPU6050_PWR_MGMT_1              0x6B     //电源管理1
#define MPU6050_PWR_MGMT_2              0x6C     //电源管理2
#define MPU6050_WHO_AM_I                0x75     //设备ID

#define BYTE_TO_HALFWORD(high,low) (int16_t)(high << 8 | low)

// 线性标度变换
static float scale_transform(float Sample_Value, float URV, float LRV)
{
    float Data;             //定义用来保存变换后的数据变量
    float Value_L = -32767.0; //定义采样值下限变量   MPU6050寄存器是16位的，最高位是符号位，
    float Value_U = 32767.0;  //定义采样值上限变量   所以寄存器输出范围是-7FFF~7FFF,对应十进制-32767~32767
    
    /* 公式：当前数据 =（采样值 - 采样值下限）/（采样值上限 - 采样值下限）*（量程上限 - 量程下限）+ 量程下限     */
    Data = (Sample_Value - Value_L) / (Value_U - Value_L) * (URV - LRV) + LRV;
    return Data;
}

static esp_err_t mpu6050_write(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t mpu6050_read(i2c_port_t i2c_num, uint8_t reg_addr, uint8_t *data, size_t data_len) {
    if (data_len == 0) {
        return ESP_OK;
    }
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDRESS << 1) | I2C_MASTER_READ, true);
    if (data_len > 1) {
        i2c_master_read(cmd, data, data_len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + data_len - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_num, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static uint8_t mpu6050_get_id(i2c_port_t i2c_num) {
    uint8_t id;
    mpu6050_read(i2c_num,MPU6050_WHO_AM_I,&id,sizeof(uint8_t));
    return id;
}

bool mpu6050_init(i2c_port_t i2c_num) {
    uint8_t id = mpu6050_get_id(i2c_num);
    if(id == 0x68) {
        printf("found mpu6050\n");
    }
    else {
        printf("can't find mpu6050, using address = 0x%x, got id = 0x%x\n",MPU6050_ADDRESS,id);
        return false;
    }

    mpu6050_write(i2c_num,MPU6050_PWR_MGMT_1,0x01);
    mpu6050_write(i2c_num,MPU6050_PWR_MGMT_2, 0x00);   //电源管理2寄存器，保持默认值，所有轴不休眠。
    mpu6050_write(i2c_num,MPU6050_SMPLRT_DIV, 0x09);   //采样率分频寄存器，
    mpu6050_write(i2c_num,MPU6050_CONFIG, 0x06);       //配置寄存器，数字低通滤波器的带宽为5Hz，陀螺仪的延迟为19.0ms
    mpu6050_write(i2c_num,MPU6050_GYRO_CONFIG, 0x18);  //陀螺仪配置寄存器，选择满量程 ±2000°/s
    mpu6050_write(i2c_num,MPU6050_ACCEL_CONFIG, 0x18); //加速度计配置寄存器，选择满量程 ±16g

    return true;
}

void mpu6050_get_accel(i2c_port_t i2c_num,int16_t* accel_array) {
    // 高低8位
    uint8_t data_l;
    uint8_t data_h;
 
    /*读取加速度计原始数值，寄存器地址含义需要翻手册的描述*/
    
    // 读X轴高低两位
    mpu6050_read(i2c_num,MPU6050_ACCEL_XOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_ACCEL_XOUT_H,&data_h,sizeof(uint8_t));
    accel_array[0] = BYTE_TO_HALFWORD(data_h,data_l);
    
    // 读Y轴高低两位
    mpu6050_read(i2c_num,MPU6050_ACCEL_YOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_ACCEL_YOUT_H,&data_h,sizeof(uint8_t));
    accel_array[1] = BYTE_TO_HALFWORD(data_h,data_l);
    
    // 读Z轴高低两位
    mpu6050_read(i2c_num,MPU6050_ACCEL_ZOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_ACCEL_ZOUT_H,&data_h,sizeof(uint8_t));
    accel_array[2] = BYTE_TO_HALFWORD(data_h,data_l);
}

void mpu6050_get_gyro(i2c_port_t i2c_num,int16_t* gyro_array) {
    // 高低8位
    uint8_t data_l;
    uint8_t data_h;
 
    // 读X轴高低两位
    mpu6050_read(i2c_num,MPU6050_GYRO_XOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_GYRO_XOUT_H,&data_h,sizeof(uint8_t));
    gyro_array[0] = BYTE_TO_HALFWORD(data_h,data_l);

    // 读Y轴高低两位
    mpu6050_read(i2c_num,MPU6050_GYRO_YOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_GYRO_YOUT_H,&data_h,sizeof(uint8_t));
    gyro_array[0] = BYTE_TO_HALFWORD(data_h,data_l);

    // 读Z轴高低两位
    mpu6050_read(i2c_num,MPU6050_GYRO_ZOUT_L,&data_l,sizeof(uint8_t));
    mpu6050_read(i2c_num,MPU6050_GYRO_ZOUT_H,&data_h,sizeof(uint8_t));
    gyro_array[0] = BYTE_TO_HALFWORD(data_h,data_l);
}

void mpu6050_get_accel_val(i2c_port_t i2c_num,float* accel_value) {
    int16_t accel_array[3];
    mpu6050_get_accel(i2c_num,accel_array);   
    
    accel_value[0] = scale_transform( (float)accel_array[0], 16.0, -16.0);  //转换X轴
    accel_value[1] = scale_transform( (float)accel_array[1], 16.0, -16.0);  //转换Y轴
    accel_value[2] = scale_transform( (float)accel_array[2], 16.0, -16.0);  //转换Z轴
}

void mpu6050_get_gyro_val(i2c_port_t i2c_num,float* gyro_value) {
    int16_t gyro_array[3];        
    mpu6050_get_gyro(i2c_num,gyro_array);  
    
    /*开始转换陀螺仪值*/
    gyro_value[0] = scale_transform( (float)gyro_array[0], 2000.0, -2000.0);  //转换X轴
    gyro_value[1] = scale_transform( (float)gyro_array[1], 2000.0, -2000.0);  //转换Y轴
    gyro_value[2] = scale_transform( (float)gyro_array[2], 2000.0, -2000.0);  //转换Z轴
}

void mpu6050_get_temperature(i2c_port_t i2c_num,float* temperature) {
    uint8_t buffer[2];
    mpu6050_read(i2c_num, MPU6050_TEMP_OUT_H, buffer, 2);
    *temperature = BYTE_TO_HALFWORD(buffer[0],buffer[1]) / 340.0 + 36.53;;
}

void mpu6050_kalman_init(mpu6050_kalman_state_t* state, float angle, float bias, float rate) {
    state->angle = angle;
    state->bias = bias;
    state->rate = rate;
    state->P[0][0] = 0.0;
    state->P[0][1] = 0.0;
    state->P[1][0] = 0.0;
    state->P[1][1] = 0.0;
}

static const float Q_gyro = 0.5;
static const float Q_angle = 0.5;
static const float R_angle = 0.5;

void mpu6050_kalman_update(mpu6050_kalman_state_t* state, float gyro_rate, float accel_angle, float dt) {
    // 预测步骤
    state->rate = gyro_rate - state->bias;
    state->angle += dt * state->rate;

    state->P[0][0] += dt * (dt*state->P[1][1] - state->P[0][1] - state->P[1][0] + Q_angle);
    state->P[0][1] -= dt * state->P[1][1];
    state->P[1][0] -= dt * state->P[1][1];
    state->P[1][1] += Q_gyro * dt;

    // 更新步骤
    float y = accel_angle - state->angle;  // 角度误差
    float S = state->P[0][0] + R_angle;  // 误差协方差
    float K[2];  // 卡尔曼增益
    K[0] = state->P[0][0] / S;
    K[1] = state->P[1][0] / S;

    state->angle += K[0] * y;
    state->bias += K[1] * y;
    float P00_temp = state->P[0][0];
    float P01_temp = state->P[0][1];

    state->P[0][0] -= K[0] * P00_temp;
    state->P[0][1] -= K[0] * P01_temp;
    state->P[1][0] -= K[1] * P00_temp;
    state->P[1][1] -= K[1] * P01_temp;
}