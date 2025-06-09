/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <cmsis_os2.h>
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */
extern ADC_HandleTypeDef hadc1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim3;
extern TIM_HandleTypeDef htim4;
extern TIM_HandleTypeDef htim5;
extern UART_HandleTypeDef huart1;
extern osSemaphoreId_t it_timer_sem;
extern osSemaphoreId_t sensor_timer_sem;
extern osMutexId_t ultra_sonic_mutex;
//extern osEventFlagsId_t drive_flags;
extern osMessageQueueId_t led_message_queue;
extern osMessageQueueId_t drive_message_queue;
extern osMessageQueueId_t manual_drive_message_queue;
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define MOTOR_EN_A_Pin GPIO_PIN_0
#define MOTOR_EN_A_GPIO_Port GPIOA
#define MOTOR_INPUT3_Pin GPIO_PIN_1
#define MOTOR_INPUT3_GPIO_Port GPIOA
#define MOTOR_INPUT1_Pin GPIO_PIN_2
#define MOTOR_INPUT1_GPIO_Port GPIOA
#define MOTOR_EN_B_Pin GPIO_PIN_3
#define MOTOR_EN_B_GPIO_Port GPIOA
#define KEY2_Pin GPIO_PIN_5
#define KEY2_GPIO_Port GPIOA
#define KEY1_Pin GPIO_PIN_5
#define KEY1_GPIO_Port GPIOC
#define IR_SWITCH_3_Pin GPIO_PIN_12
#define IR_SWITCH_3_GPIO_Port GPIOB
#define IR_SWITCH_2_Pin GPIO_PIN_14
#define IR_SWITCH_2_GPIO_Port GPIOB
#define IR_SWITCH_1_Pin GPIO_PIN_15
#define IR_SWITCH_1_GPIO_Port GPIOB
#define IR_SWITCH_4_Pin GPIO_PIN_6
#define IR_SWITCH_4_GPIO_Port GPIOC
#define IR_SWITCH_5_Pin GPIO_PIN_7
#define IR_SWITCH_5_GPIO_Port GPIOC
#define MOTOR_INPUT4_Pin GPIO_PIN_9
#define MOTOR_INPUT4_GPIO_Port GPIOC
#define ULTRA_SONIC_ECHO_Pin GPIO_PIN_4
#define ULTRA_SONIC_ECHO_GPIO_Port GPIOB
#define ULTRA_SONIC_ECHO_EXTI_IRQn EXTI4_IRQn
#define ULTRA_SONIC_TRIG_Pin GPIO_PIN_5
#define ULTRA_SONIC_TRIG_GPIO_Port GPIOB
#define LED_G_Pin GPIO_PIN_6
#define LED_G_GPIO_Port GPIOB
#define LED_B_Pin GPIO_PIN_7
#define LED_B_GPIO_Port GPIOB
#define LED_R_Pin GPIO_PIN_8
#define LED_R_GPIO_Port GPIOB
#define MOTOR_INPUT2_Pin GPIO_PIN_9
#define MOTOR_INPUT2_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
