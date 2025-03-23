#include <stdio.h>
#include "Application/init.h"
#include "Application/test_led.h"
#include "Application/oled.h"
#include "Application/balance.h"

#define TASK_NAME "init"

static const osThreadAttr_t taskAttributes = {
    .name = TASK_NAME,
    .stack_size = 128 * 4,
    .priority = (osPriority_t) osPriorityAboveNormal,
};

osThreadId_t initTaskHandle;
#ifdef USE_MPU_DMP
osSemaphoreId_t mpu6050_semaphore, mpu6050_inited_semaphore;
#endif

static void task(void *argument) {
#ifdef USE_MPU_DMP
    mpu6050_semaphore = osSemaphoreNew(1,0,NULL);
    mpu6050_inited_semaphore = osSemaphoreNew(1,0,NULL);
#endif
    // start up all threads then quit
    testLEDTaskLaunch();
    oledTaskLaunch();
    balanceTaskLaunch();
    osThreadExit();
}

void initTaskLaunch(void) {
    printf("launching task \"%s\"\n",TASK_NAME);
    initTaskHandle = osThreadNew(task,NULL,&taskAttributes);
}