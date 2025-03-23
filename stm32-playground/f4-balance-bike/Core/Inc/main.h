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

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

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
#define WIRELESS_CE_Pin GPIO_PIN_8
#define WIRELESS_CE_GPIO_Port GPIOE
#define WIRELESS_CSN_Pin GPIO_PIN_9
#define WIRELESS_CSN_GPIO_Port GPIOE
#define TEST_LED_Pin GPIO_PIN_12
#define TEST_LED_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
