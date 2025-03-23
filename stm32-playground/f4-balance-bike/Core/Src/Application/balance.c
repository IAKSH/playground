#include <stdio.h>
#include <math.h>
#include "Application/balance.h"

#ifdef USE_MPU_DMP
#include "Driver/mpu6050_dmp/mpu6050_dmp.h"
#else
#include "Driver/mpu6050/mpu6050.h"
#endif

#include "Driver/motor/motor.h"
#include "Utils/abs.h"
#include "Utils/pid.h"

#define TASK_NAME "balance"

// 任务属性
static const osThreadAttr_t taskAttributes = {
    .name = TASK_NAME,
    .stack_size = 128 * 8,
    .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t balanceTaskHandle;

#ifndef USE_MPU_DMP
MPU6050_t mpu6050;
#else
extern osSemaphoreId_t mpu6050_inited_semaphore;
#endif
float motorSpeedA, motorSpeedB;
extern osSemaphoreId_t mpu6050_semaphore;

static uint16_t motorOutput;

// 内环：平衡 PID
static PID_TypeDef pitchPID;
static double pitch;              // 当前倾角（来自 MPU6050）
double pitchPIDOutput, pitchPIDSetpoint;  // 平衡 PID 的输出和设定值

// 外环1：速度 PID
static PID_TypeDef speedPID;
//static double measuredSpeed;      // 车速测量（两个轮子平均速度）
double measuredSpeed;
double speedPIDOutput, speedPIDSetpoint;   // 速度 PID 输出和设定值

// 外环2：转向 PID
static PID_TypeDef turningPID;
static double turning;            // 转向反馈值（例如：左右轮速度之差）
double turningPIDOutput, turningPIDSetpoint;  // 转向 PID 输出和设定值

extern I2C_HandleTypeDef hi2c1;

// 假设目标速度与转向由上层控制（这里简单初始化为0）
double targetSpeed = 0.0;    // 正值：前进；负值：后退
double targetTurning = 0.0;  // 正值：右转；负值：左转

#ifdef USE_MPU_DMP
float mpu6050_pitch,mpu_6050_roll,mpu_6050_yaw;

static const osThreadAttr_t mpuDmpTaskAttributes = {
    .name = TASK_NAME,
    .stack_size = 128 * 8,
    .priority = (osPriority_t) osPriorityNormal1,
};

osThreadId_t mpuDmpTaskHandle;
int mpu6050_init_ret;

static void mpuDmpTask(void* arg) {
    while(mpu6050_init_ret = MPU6050_DMP_init(),mpu6050_init_ret != 0) {
        printf("MPU 6050 init failed, ret = %d\n",mpu6050_init_ret);
        osDelay(100);
    }
    osSemaphoreRelease(mpu6050_inited_semaphore);
    while(1) {
        osSemaphoreAcquire(mpu6050_semaphore, osWaitForever);
        MPU6050_DMP_Get_Date(&mpu6050_pitch,&mpu_6050_roll,&mpu_6050_yaw);
    }
}
#endif

static void task(void* arg) {
#ifndef USE_MPU_DMP
    // 初始化 MPU6050，直到初始化成功
    while (MPU6050_Init(&hi2c1) == 1) {
        printf("mpu6050 init failed\n");
        osDelay(250);
    }
#endif
    // 初始化电机
    motorInit();
    motorSetDirect(MOTOR_A, MOTOR_FORWARD);
    motorSetDirect(MOTOR_B, MOTOR_FORWARD);

    // ----- 初始化内环：平衡 PID ----- //
    // 设定参数：这里取原来比较合适的参数，设定输出限幅为 -500+64 ~ 500-64
    PID(&pitchPID, &pitch, &pitchPIDOutput, &pitchPIDSetpoint,
        150, 0, 6, _PID_P_ON_E, _PID_CD_DIRECT);          
    PID_SetMode(&pitchPID, _PID_MODE_AUTOMATIC); 
    PID_SetOutputLimits(&pitchPID, -7400+10, 7400-10);

    // ----- 初始化外环1：速度 PID ----- //
    // 速度测量值 measuredSpeed 为两个电机的平均速度
    // 参数需要根据实际情况调整
    PID(&speedPID, &measuredSpeed, &speedPIDOutput, &speedPIDSetpoint,
        0.5, 50, 0, _PID_P_ON_E, _PID_CD_DIRECT);
    PID_SetMode(&speedPID, _PID_MODE_AUTOMATIC);
    // 限制速度 PID 输出在一个合理的倾角范围内（单位：度）
    PID_SetOutputLimits(&speedPID, -10, 10);

    // ----- 初始化外环2：转向 PID ----- //
    // turning 这里采用左右轮速度差（speedA - speedB）作为反馈,参数需调试
    PID(&turningPID, &turning, &turningPIDOutput, &turningPIDSetpoint,
        0, 0, 0, _PID_P_ON_E, _PID_CD_DIRECT);
    PID_SetMode(&turningPID, _PID_MODE_AUTOMATIC);
    // 限制转向 PID 输出，单位与内环相同（与电机速度成正比）
    PID_SetOutputLimits(&turningPID, -100, 100);

    // 初始设定值
    pitchPIDSetpoint = 0.0;         // 初始内环平衡目标：保持竖直
    speedPIDSetpoint = targetSpeed; // 外环速度目标来自目标输入
    turningPIDSetpoint = targetTurning; // 外环转向目标

    while (1) {
#ifndef USE_MPU_DMP
        MPU6050_Read_All(&hi2c1, &mpu6050);
        // 注意：此处对 KalmanAngleY 进行了偏移和反向处理，与你的调试有关
        mpu6050.KalmanAngleY = -mpu6050.KalmanAngleY - 1;
#endif

        // 更新电机当前速度（这里 motorUpdateSpeed 需自行实现或调用已有接口）
        motorUpdateSpeed(&motorSpeedA, &motorSpeedB);

        // 防跌落保护，当倾角超出安全范围时停止电机
#ifdef USE_MPU_DMP
        if(fabs(mpu6050_pitch >= 30.0f)) {
#else
        if (fabs(mpu6050.KalmanAngleY) >= 30.0f) {
#endif
            motorSetDirect(MOTOR_A, MOTOR_STOP);
            motorSetDirect(MOTOR_B, MOTOR_STOP);
            osDelay(1);
            continue;
        }

        // 数据转换
#ifdef USE_MPU_DMP
        pitch = mpu6050_pitch;
#else
        pitch = mpu6050.KalmanAngleY;
#endif
        
        // 用两个轮子速度的平均值作为车速测量值
        measuredSpeed = (motorSpeedA + motorSpeedB) / 2.0;;
        // 计算转向反馈（这里简单采用左右轮速度差，实际中可换成陀螺仪 Z 轴数据）
        turning = motorSpeedA - motorSpeedB;

        // ----- 外环速度 PID 处理 ----- //
        // 更新速度目标（如果外部改变 targetSpeed，此处自动更新）
        speedPIDSetpoint = targetSpeed;
        PID_Compute(&speedPID);
        // 将速度环输出作为内环平衡目标（即期望倾角），正值表示向前倾，从而加速前进
        pitchPIDSetpoint = speedPIDOutput;

        // ----- 外环转向 PID 处理 ----- //
        turningPIDSetpoint = targetTurning;
        PID_Compute(&turningPID);

        // ----- 内环平衡 PID 处理 ----- //
        PID_Compute(&pitchPID);

        // 最终电机控制量 = 平衡内环输出 ± 转向修正
        double motorOutputA = pitchPIDOutput + turningPIDOutput;
        double motorOutputB = pitchPIDOutput - turningPIDOutput;

        // 根据控制量方向设置电机转动方向
        motorSetDirect(MOTOR_A, (motorOutputA < 0) ? MOTOR_BACKWARD : MOTOR_FORWARD);
        motorSetDirect(MOTOR_B, (motorOutputB < 0) ? MOTOR_BACKWARD : MOTOR_FORWARD);

        // 将输出量取绝对值后作为 PWM 输出，若输出小于一定值（例如 20），则补偿以克服摩擦阻尼
        uint16_t outputA = (uint16_t)fabs(motorOutputA);
        if (fabs(motorOutputA) > 0 && outputA < 30)
            outputA += 30;
        motorSetSpeed(MOTOR_A, outputA);

        uint16_t outputB = (uint16_t)fabs(motorOutputB);
        if (fabs(motorOutputB) > 0 && outputB < 30)
            outputB += 30;
        motorSetSpeed(MOTOR_B, outputB);

#ifndef USE_MPU_DMP
        osDelay(1);
#endif
    }
}

void balanceTaskLaunch(void) {
    printf("launching task \"%s\"\n", TASK_NAME);
    balanceTaskHandle = osThreadNew(task, NULL, &taskAttributes);
#ifdef USE_MPU_DMP
    mpuDmpTaskHandle = osThreadNew(mpuDmpTask,NULL,&mpuDmpTaskAttributes);
#endif
}
