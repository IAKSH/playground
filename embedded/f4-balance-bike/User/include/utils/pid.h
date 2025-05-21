#pragma once

/* PID 控制器结构体定义 */
typedef struct {
    float Kp;        // 比例系数
    float Ki;        // 积分系数
    float Kd;        // 微分系数
    float setpoint;  // 目标值
    float lastError; // 上一次误差值
    float integral;  // 积分累积项
    float output;    // 当前输出值
} PID;

/* PID 计算函数：传入当前测量值和采样周期 dt */
float pid_compute(PID *pid, float measurement, float dt);