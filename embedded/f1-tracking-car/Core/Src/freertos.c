/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
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

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "cmsis_os2.h"
#include "tasks.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */
osThreadId_t drive_task_handle;
const osThreadAttr_t drive_task_attributes = {
  .name = "drive",
  .stack_size = 128 * 4,
  .priority = (osPriority_t)osPriorityNormal
};

osThreadId_t led_task_handle;
const osThreadAttr_t led_task_attributes = {
  .name = "led",
  .stack_size = 128 * 4,
  .priority = (osPriority_t)osPriorityBelowNormal
};

osThreadId_t control_task_handle;
const osThreadAttr_t control_task_attributes = {
  .name = "control",
  .stack_size = 128 * 4,
  .priority = (osPriority_t)osPriorityBelowNormal
};

osThreadId_t sensor_task_handle;
const osThreadAttr_t sensor_task_attributes = {
  .name = "sensor",
  .stack_size = 128 * 4,
  .priority = (osPriority_t)osPriorityNormal
};

osSemaphoreId_t it_timer_sem;
osSemaphoreId_t sensor_timer_sem;
osMutexId_t ultra_sonic_mutex;
//osEventFlagsId_t drive_flags;
osMessageQueueId_t led_message_queue;
osMessageQueueId_t drive_message_queue;
osMessageQueueId_t manual_drive_message_queue;
/* USER CODE END Variables */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */
void create_thread(void) {
  // semaphores
  it_timer_sem = osSemaphoreNew(1,0,NULL);
  sensor_timer_sem = osSemaphoreNew(1,0,NULL);

  // mutex
  ultra_sonic_mutex = osMutexNew(NULL);

  // event flags
  //drive_flags = osEventFlagsNew(NULL);

  // message queue
  led_message_queue = osMessageQueueNew(16,sizeof(int),NULL);
  drive_message_queue = osMessageQueueNew(16,sizeof(int),NULL);
  manual_drive_message_queue = osMessageQueueNew(16,sizeof(int),NULL);

  // threads
  drive_task_handle = osThreadNew(drive_task,NULL,&drive_task_attributes);
  led_task_handle = osThreadNew(led_task,NULL,&led_task_attributes);
  control_task_handle = osThreadNew(control_task,NULL,&control_task_attributes);
  sensor_task_handle = osThreadNew(sensor_task,NULL,&sensor_task_attributes);

  // misc
  __HAL_TIM_CLEAR_IT(&htim3, TIM_IT_UPDATE);	
  HAL_TIM_Base_Start_IT(&htim3);
}
/* USER CODE END FunctionPrototypes */

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

