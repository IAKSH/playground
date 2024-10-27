#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "../tasks.h"
#include "../../drone_status.h"
#include <motor.h>
#include <math.h>

static float integral[3] = {0.0, 0.0, 0.0};
static float previous_error[3] = {0.0, 0.0, 0.0};
static const float integral_limit = 10.0;
static const float derivative_limit = 20.0;  // 防止导数项过度反应
static const float filter_coefficient = 0.5; // 滤波系数

static void pid_control() {
    float error[3];
    float derivative[3];
    float output[3];

    for (int i = 0; i < 3; i++) {
        error[i] = drone_target_euler[i] - drone_gryo_euler[i];

        if (fabs(error[i]) < integral_limit) {
            integral[i] += error[i];
        }
        if (integral[i] > integral_limit) integral[i] = integral_limit;
        else if (integral[i] < -integral_limit) integral[i] = -integral_limit;

        derivative[i] = error[i] - previous_error[i];
        if (derivative[i] > derivative_limit) derivative[i] = derivative_limit;
        else if (derivative[i] < -derivative_limit) derivative[i] = -derivative_limit;

        output[i] = drone_motor_kp * error[i] + drone_motor_ki * integral[i] + drone_motor_kd * derivative[i];

        previous_error[i] = error[i];
    }

    // 添加低通滤波器
    float filtered_output[3];
    for (int i = 0; i < 3; i++) {
        filtered_output[i] = filter_coefficient * output[i] + (1 - filter_coefficient) * previous_error[i];
    }

    drone_motor_duty[1] = (drone_motor_offset + filtered_output[0] + filtered_output[1] - filtered_output[2]);// 前左
    drone_motor_duty[2] = (drone_motor_offset - filtered_output[0] + filtered_output[1] + filtered_output[2]);// 前右
    drone_motor_duty[3] = (drone_motor_offset + filtered_output[0] - filtered_output[1] + filtered_output[2]);// 后右
    drone_motor_duty[4] = (drone_motor_offset - filtered_output[0] - filtered_output[1] - filtered_output[2]);// 后左

    for (int i = 1; i < 5; i++) {
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
        vTaskDelay(pdMS_TO_TICKS(16));
    }
    vTaskDelete(NULL);
}
