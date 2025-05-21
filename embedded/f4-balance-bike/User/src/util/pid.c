#include "utils/pid.h"

float pid_compute(PID *pid, float measurement, float dt) {
    float error = pid->setpoint - measurement;
    pid->integral += error * dt;

    // 积分限幅，防止积分风up
    if(pid->integral > 1000.0f) pid->integral = 1000.0f;
    if(pid->integral < -1000.0f) pid->integral = -1000.0f;

    float derivative = (error - pid->lastError) / dt;
    pid->output = pid->Kp * error + pid->Ki * pid->integral + pid->Kd * derivative;
    pid->lastError = error;
    return pid->output;
}