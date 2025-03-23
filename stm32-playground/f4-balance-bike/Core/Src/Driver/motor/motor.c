#include "Driver/motor/motor.h"

void motorInit(void) {
    HAL_TIM_PWM_Start(&MOTOR_PWM_TIMER,TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&MOTOR_PWM_TIMER,TIM_CHANNEL_4);

    HAL_TIM_Encoder_Start(&MOTOR_A_ENCODER_TIMER,TIM_CHANNEL_ALL);
    HAL_TIM_Encoder_Start(&MOTOR_B_ENCODER_TIMER,TIM_CHANNEL_ALL);
    HAL_TIM_Base_Start_IT(&MOTOR_A_ENCODER_TIMER);
    HAL_TIM_Base_Start_IT(&MOTOR_B_ENCODER_TIMER);
}

void motorSetDirect(MotorID id, MotorDirection direction) {
    switch(id) {
        case MOTOR_A:
            HAL_GPIO_WritePin(MOTOR_CONTROL_A1_PORT,MOTOR_CONTROL_A1_PIN,direction == MOTOR_BACKWARD);
            HAL_GPIO_WritePin(MOTOR_CONTROL_A2_PORT,MOTOR_CONTROL_A2_PIN,direction == MOTOR_FORWARD);
            break;
        case MOTOR_B:
            HAL_GPIO_WritePin(MOTOR_CONTROL_B1_PORT,MOTOR_CONTROL_B1_PIN,direction == MOTOR_FORWARD);
            HAL_GPIO_WritePin(MOTOR_CONTROL_B2_PORT,MOTOR_CONTROL_B2_PIN,direction == MOTOR_BACKWARD);
            break;
        default:
            break;
    }
}

void motorSetSpeed(MotorID id,uint16_t speed) {
    switch(id) {
        case MOTOR_A:
            __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, TIM_CHANNEL_3, speed);
            break;
        case MOTOR_B:
            __HAL_TIM_SET_COMPARE(&MOTOR_PWM_TIMER, TIM_CHANNEL_4, speed);
            break;
        default:
            break;
    }
}

void motorUpdateSpeed(float* motorSpeedA,float* motorBSpeedB) { 
    //*motorSpeedA = (float)__HAL_TIM_GET_COUNTER(&htim2) * 100.0f / 9.6f / 11.0f / 4.0f;
    //*motorBSpeedB = (float)__HAL_TIM_GET_COUNTER(&htim4) * 100.0f / 9.6f / 11.0f / 4.0f;
    static uint16_t speed[4];
    speed[0] = (uint16_t)__HAL_TIM_GET_COUNTER(&htim2);
    speed[1] = (uint16_t)__HAL_TIM_GET_COUNTER(&htim4);
    speed[2] = UINT16_MAX - speed[0];
    speed[3] = UINT16_MAX - speed[1];

    *motorSpeedA = speed[0] < speed[2] ? speed[0] : -(float)speed[2];
    *motorBSpeedB = speed[1] < speed[3] ? speed[1] : -(float)speed[3];
    *motorBSpeedB = -*motorBSpeedB;
    
    __HAL_TIM_SET_COUNTER(&htim2,0);
    __HAL_TIM_SET_COUNTER(&htim4,0);
}   
    