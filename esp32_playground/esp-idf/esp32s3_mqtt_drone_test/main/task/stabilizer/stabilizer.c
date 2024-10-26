#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "../tasks.h"
#include "../../drone_status.h"
#include <motor.h>

// PID控制器误差累积
static float integral[3] = {0.0, 0.0, 0.0};
static float previous_error[3] = {0.0, 0.0, 0.0};
static float previous_derivative[3] = {0.0, 0.0, 0.0};
static const float integral_limit = 10.0;
static const float filter_coefficient = 0.9; // 0到1之间，接近1时滤波效果强

static void pid_control() {
    float error[3];
    float derivative[3];
    float output[3];

    for (int i = 0; i < 3; i++) {
        error[i] = drone_target_euler[i] - drone_gryo_euler[i];
        integral[i] += error[i];
        if (integral[i] > integral_limit) integral[i] = integral_limit;
        else if (integral[i] < -integral_limit) integral[i] = -integral_limit;
        derivative[i] = filter_coefficient * previous_derivative[i] + (1.0 - filter_coefficient) * (error[i] - previous_error[i]);
        previous_derivative[i] = derivative[i];
        previous_error[i] = error[i];
        output[i] = drone_motor_kp * error[i] + drone_motor_ki * integral[i] + drone_motor_kd * derivative[i];
    }

    drone_motor_duty[1] = (uint16_t)(drone_motor_offset + output[0] + output[1]);
    drone_motor_duty[2] = (uint16_t)(drone_motor_offset + output[0] - output[1]);
    drone_motor_duty[3] = (uint16_t)(drone_motor_offset - output[0] + output[1]);
    drone_motor_duty[4] = (uint16_t)(drone_motor_offset - output[0] - output[1]);

    for (int i = 0; i < 5; i++) {
        if (drone_motor_duty[i] > 8192) {
            drone_motor_duty[i] = 8192;
            integral[i] -= error[i]; // 反馈调整积分项
        } else if (drone_motor_duty[i] < 500) {
            drone_motor_duty[i] = 500;
            integral[i] -= error[i]; // 反馈调整积分项
        }
        set_motor_duty(i, drone_motor_duty[i]);
    }
}

void task_stabilizer(void) {
    motor_ledc_initialize();
    while (1) {
        if(!drone_motor_emergency_stop)
            pid_control();
        else {
            for(int i = 0;i < 5;i++)
                set_motor_duty(i,0);
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    vTaskDelete(NULL);
}
