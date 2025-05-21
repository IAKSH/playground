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
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "cmsis_os2.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */
extern osThreadId_t defaultTaskHandle;
extern const osThreadAttr_t defaultTask_attributes;
extern osThreadId_t test_led_task_handle;
extern const osThreadAttr_t test_led_task_attributes;
extern osThreadId_t oled_task_handle;
extern const osThreadAttr_t oled_task_attributes;
extern osThreadId_t balance_task_handle;
extern const osThreadAttr_t balance_task_attributes;
extern osSemaphoreId_t gyro_ready_sem;
extern osMutexId_t i2c_bus_mutex;
extern osEventFlagsId_t event;
extern osMessageQueueId_t command_queue;

extern I2C_HandleTypeDef hi2c1;
extern SPI_HandleTypeDef hspi2;
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
#define MOTOR_IO1_Pin GPIO_PIN_2
#define MOTOR_IO1_GPIO_Port GPIOE
#define MOTOR_IO2_Pin GPIO_PIN_3
#define MOTOR_IO2_GPIO_Port GPIOE
#define MOTOR_IO3_Pin GPIO_PIN_0
#define MOTOR_IO3_GPIO_Port GPIOC
#define MOTOR_IO4_Pin GPIO_PIN_1
#define MOTOR_IO4_GPIO_Port GPIOC
#define EXTI2_MPU_Pin GPIO_PIN_2
#define EXTI2_MPU_GPIO_Port GPIOA
#define EXTI2_MPU_EXTI_IRQn EXTI2_IRQn
#define EXIT7_WIRELESS_IRQ_Pin GPIO_PIN_7
#define EXIT7_WIRELESS_IRQ_GPIO_Port GPIOE
#define EXIT7_WIRELESS_IRQ_EXTI_IRQn EXTI9_5_IRQn
#define WIRELESS_CE_Pin GPIO_PIN_8
#define WIRELESS_CE_GPIO_Port GPIOE
#define WIRELESS_CSN_Pin GPIO_PIN_9
#define WIRELESS_CSN_GPIO_Port GPIOE
#define TEST_LED_Pin GPIO_PIN_12
#define TEST_LED_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */
#define EVENT_FLAG_GYRO_INITIALIZED 0x01
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
