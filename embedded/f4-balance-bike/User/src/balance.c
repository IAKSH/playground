#include "cmsis_os2.h"
#include "driver/motor.h"
#include "driver/gyro.h"
#include <math.h>
#include <stdio.h>

float mpu6050_pitch,mpu_6050_roll,mpu_6050_yaw;
float motorSpeedA,motorSpeedB;

/* PID 控制器结构体定义 */
typedef struct {
    float Kp;        // 比例系数
    float Ki;        // 积分系数
    float Kd;        // 微分系数
    float setpoint;  // 目标值
    float lastError; // 上一次误差值
    float integral;  // 积分累积项
    float output;    // 当前输出值
} PID_Controller;

/* PID 计算函数：传入当前测量值和采样周期 dt */
static float PID_Compute(PID_Controller *pid, float measurement, float dt) {
    float error = pid->setpoint - measurement;
    pid->integral += error * dt;
    float derivative = (error - pid->lastError) / dt;
    pid->output = pid->Kp * error + pid->Ki * pid->integral + pid->Kd * derivative;
    pid->lastError = error;
    return pid->output;
}

#define ANGLE_KP 275.0f
#define ANGLE_KI 0.0f
#define ANGLE_KD 10.0f 

#define SPEED_KP 1.8f
#define SPEED_KI 0.0f
#define SPEED_KD 0.0f

/* 初始化 PID 参数，此处参数需要根据系统实际情况做具体调试 */ 
static PID_Controller pid_angle  = {
    .Kp = ANGLE_KP, .Ki = ANGLE_KI, .Kd =ANGLE_KD,      // 外环：车体倾角控制
    .setpoint = 0.0f,
    .lastError = 0.0f,
    .integral = 0.0f,
    .output = 0.0f
};

static PID_Controller pid_speed_A = {
    .Kp = SPEED_KP, .Ki = SPEED_KI, .Kd = SPEED_KD,        // 内环：左轮速度控制
    .setpoint = 0.0f,
    .lastError = 0.0f,
    .integral = 0.0f,
    .output = 0.0f
};

static PID_Controller pid_speed_B = {
    .Kp = SPEED_KP, .Ki = SPEED_KI, .Kd =SPEED_KD,        // 内环：右轮速度控制
    .setpoint = 0.0f,
    .lastError = 0.0f,
    .integral = 0.0f,
    .output = 0.0f
};

/* 修改后的任务函数，添加了串级 PID 控制实现 */
void balance_task(void* arg) {
    int mpu6050_init_ret;
    while((mpu6050_init_ret = gyro_init()) != 0) {
        printf("MPU6050 init failed, ret = %d\n", mpu6050_init_ret);
        osDelay(100);
    }
    osEventFlagsSet(event,EVENT_FLAG_GYRO_INITIALIZED);

    // 电机初始化以及设置初始旋转方向
    motorInit();
    motorSetDirect(MOTOR_A, MOTOR_FORWARD);
    motorSetDirect(MOTOR_B, MOTOR_FORWARD);

    while (1) {
        // 获取 MPU 数据
        osSemaphoreAcquire(gyro_ready_sem, osWaitForever);
        gyro_get_data(&mpu6050_pitch, &mpu_6050_roll, &mpu_6050_yaw);

        // 更新电机速度（motorSpeedA 与 motorSpeedB 分别为两个轮子的测速值）
        motorUpdateSpeed(&motorSpeedA, &motorSpeedB);

        // 防跌落保护：当车体倾角过大时直接停止电机
        if (fabs(mpu6050_pitch) >= 40.0f) {
            motorSetDirect(MOTOR_A, MOTOR_STOP);
            motorSetDirect(MOTOR_B, MOTOR_STOP);
            continue;
        }

        // 设定采样周期（dt，单位：秒），该值需要依据实际任务周期确定
        float dt = 0.005f;   // 此处假设 5ms 为一个采样周期

        // 读取当前车体倾角（可用 mpu6050_pitch 或 KalmanAngleY，根据使用的 MPU 模式）
        float currentAngle = 0.0f;

        currentAngle = mpu6050_pitch;

        /* 外环 PID 控制：目标倾角设为 0°（竖直状态），计算倾角修正量 */
        pid_angle.setpoint = 0.0f;
        float angleCorrection = PID_Compute(&pid_angle, currentAngle, dt);

        /* 内环 PID 控制：以外环输出作为左右轮目标速度
           注：如果系统是对称设计，两个轮子的 PID 参数可以一样 */
        pid_speed_A.setpoint = angleCorrection;
        pid_speed_B.setpoint = angleCorrection;
        float speedCommandA = PID_Compute(&pid_speed_A, motorSpeedA, dt);
        float speedCommandB = PID_Compute(&pid_speed_B, motorSpeedB, dt);

        /* 根据 PID 输出决定电机旋转方向及 PWM 控制值 */
        if (speedCommandA >= 0) {
            motorSetDirect(MOTOR_A, MOTOR_FORWARD);
        } else {
            motorSetDirect(MOTOR_A, MOTOR_BACKWARD);
            speedCommandA = -speedCommandA; // 取绝对值
        }

        if (speedCommandB >= 0) {
            motorSetDirect(MOTOR_B, MOTOR_FORWARD);
        } else {
            motorSetDirect(MOTOR_B, MOTOR_BACKWARD);
            speedCommandB = -speedCommandB;
        }

        speedCommandA += speedCommandA > 0 ? 150.0f : -150.0f;
        speedCommandB += speedCommandB > 0 ? 150.0f : -150.0f;

        // 限制 PWM 输出上限（例如设为 7200），避免输出超出电机驱动范围
        if (speedCommandA > 7200.0f) speedCommandA = 7200.0f;
        if (speedCommandB > 7200.0f) speedCommandB = 7200.0f;

        motorSetSpeed(MOTOR_A, (uint16_t)speedCommandA);
        motorSetSpeed(MOTOR_B, (uint16_t)speedCommandB);
    }
}