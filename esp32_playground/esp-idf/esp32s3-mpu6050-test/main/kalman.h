#pragma once

typedef struct {
    float q;  // 过程噪声协方差
    float r;  // 测量噪声协方差
    float x;  // 估计值
    float p;  // 估计误差协方差
    float k;  // 卡尔曼增益
} KalmanState;

void kalman_init(KalmanState* state, float q, float r, float p, float initial_value);
float kalman_update(KalmanState* state, float measurement);