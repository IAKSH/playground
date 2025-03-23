#include <stdio.h>
#include "Application/test_led.h"

#define TASK_NAME "test_led"

static const osThreadAttr_t taskAttributes = {
    .name = TASK_NAME,
    .stack_size = 128,
    .priority = (osPriority_t) osPriorityBelowNormal,
};

osThreadId_t testLEDTaskHandle;

static void task(void* arg) {
    while(1) {
        HAL_GPIO_TogglePin(TEST_LED_GPIO_Port,TEST_LED_Pin);
        osDelay(500);
    }
}

void testLEDTaskLaunch(void) {
    printf("launching task \"%s\"\n",TASK_NAME);
    testLEDTaskHandle = osThreadNew(task,NULL,&taskAttributes);
}

