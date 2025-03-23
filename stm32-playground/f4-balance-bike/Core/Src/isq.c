#include <stdio.h>
#include "main.h"
#include "cmsis_os2.h"

#ifdef USE_MPU_DMP
extern osSemaphoreId_t mpu6050_semaphore;
#endif

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    switch (GPIO_Pin)
    {
#ifdef USE_MPU_DMP
    case GPIO_PIN_2:
        osSemaphoreRelease(mpu6050_semaphore);
        __HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_2);
        break;
#endif
    default:
        break;
    }
}