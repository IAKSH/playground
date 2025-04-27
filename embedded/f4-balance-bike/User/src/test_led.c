#include "main.h"
#include <stdio.h>

void test_led_task(void* arg) {
    HAL_GPIO_WritePin(TEST_LED_GPIO_Port,TEST_LED_Pin,GPIO_PIN_RESET);
    osEventFlagsWait(event,EVENT_FLAG_GYRO_INITIALIZED,osFlagsWaitAny,osWaitForever);
    while(1) {
        HAL_GPIO_TogglePin(TEST_LED_GPIO_Port,TEST_LED_Pin);
        osDelay(500);
    }
}

