#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "../tasks.h"
#include "../../drone_status.h"
#include <motor.h>

// 下面这些可以mqtt到控制中心，用来遥控

// PID控制器参数
float kp = 10.0;
float ki = 0.1;
float kd = 0.1;

// 目标姿态（例如，水平飞行）
float target_euler[3] = {0.0, 0.0, 0.0};

// PID控制器误差累积
float integral[3] = {0.0, 0.0, 0.0};
float previous_error[3] = {0.0, 0.0, 0.0};

void pid_control() {
    float error[3];
    float derivative[3];
    float output[3];

    // 计算误差
    for (int i = 0; i < 3; i++) {
        error[i] = target_euler[i] - drone_gryo_euler[i];
        integral[i] += error[i];
        derivative[i] = error[i] - previous_error[i];
        previous_error[i] = error[i];

        // PID控制器输出
        output[i] = kp * error[i] + ki * integral[i] + kd * derivative[i];
    }

    // 将PID输出转换为电机占空比
    uint16_t duties[5] = {0,0,0,0,0};

    /*
    duties[1] = (uint16_t)(4096 + output[0] + output[1]);// 左上角
    duties[2] = (uint16_t)(4096 + output[0] - output[1]);// 右上角
    duties[3] = (uint16_t)(4096 - output[0] + output[1]);// 左下角
    duties[4] = (uint16_t)(4096 - output[0] - output[1]);// 右下角
    */

    // 带偏航角
    duties[1] = (uint16_t)(4096 + output[0] + output[1] + output[2]); // 左上角
    duties[2] = (uint16_t)(4096 + output[0] - output[1] - output[2]); // 右上角
    duties[3] = (uint16_t)(4096 - output[0] + output[1] - output[2]); // 左下角
    duties[4] = (uint16_t)(4096 - output[0] - output[1] + output[2]); // 右下角

    duties[0] = (duties[1] + duties[2] + duties[3] + duties[4]) / 4;

    // 设置电机占空比
    for(int i = 0;i < 5;i++) {
        printf("%d,",duties[i]);
        set_motor_duty(i,duties[i]);
    }
    puts("\b\n");
}

void task_stabilizer(void) {
    motor_ledc_initialize();
    //motor_ledc_test(3);
    
    while(1) {
        pid_control();
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    vTaskDelete(NULL);
}
