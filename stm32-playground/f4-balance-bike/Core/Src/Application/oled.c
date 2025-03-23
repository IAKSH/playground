#include <stdio.h>
#include "Application/oled.h"
#include "Driver/ssd1306/ssd1306.h"
#include "Driver/ssd1306/ssd1306_tests.h"
#include "Driver/ssd1306/ssd1306_fonts.h"

#ifdef USE_MPU_DMP
#include "Driver/mpu6050_dmp/mpu6050_dmp.h"
#else
#include "Driver/mpu6050/mpu6050.h"
#endif

#include "Utils/abs.h"

#define TASK_NAME "oled"

static const osThreadAttr_t taskAttributes = {
    .name = TASK_NAME,
    .stack_size = 128 * 8,
    .priority = (osPriority_t) osPriorityBelowNormal,
};

osThreadId_t oledTaskHandle;

#ifdef USE_MPU_DMP
extern float mpu6050_pitch,mpu_6050_roll,mpu_6050_yaw;
extern osSemaphoreId_t mpu6050_inited_semaphore;
#else
extern MPU6050_t mpu6050;
#endif

extern double pitchPIDOutput;

static void task(void* arg) {
    ssd1306_Init();
    ssd1306_Fill(Black);

#ifdef USE_MPU_DMP
    ssd1306_SetCursor(0,0);
    ssd1306_WriteString("init MPU6050...",Font_7x10,White);
    ssd1306_UpdateScreen();
    
    osSemaphoreAcquire(mpu6050_inited_semaphore,osWaitForever);
    ssd1306_Fill(Black);
#endif

    char buf[2][32];

    while(1) {
        extern double measuredSpeed;

#ifdef USE_MPU_DMP
        snprintf(buf[0],sizeof(buf),"p= %d.%d%d r= %d.%d%d      ",
        (int)mpu6050_pitch,abs((int)(mpu6050_pitch * 10)) % 10,abs((int)(mpu6050_pitch * 100)) % 10,
        (int)mpu_6050_roll,abs((int)(mpu_6050_roll) * 10) % 10,abs((int)(mpu_6050_roll * 100)) % 10);
#else
        snprintf(buf[0],sizeof(buf),"kp= %d.%d%d p= %d.%d%d      ",
            (int)mpu6050.KalmanAngleY,abs((int)(mpu6050.KalmanAngleY * 10)) % 10,abs((int)(mpu6050.KalmanAngleY * 100)) % 10,
            (int)mpu6050.Gy,abs((int)(mpu6050.Gy) * 10) % 10,abs((int)(mpu6050.Gy * 100)) % 10);
#endif
        snprintf(buf[1],sizeof(buf),"o= %d.%d%d      ",(int)pitchPIDOutput,abs((int)(pitchPIDOutput * 10)) % 10,abs((int)(pitchPIDOutput * 100)) % 10);
        //snprintf(buf[1],sizeof(buf),"o= %d.%d%d      ",(int)measuredSpeed,abs((int)(measuredSpeed * 10)) % 10,abs((int)(measuredSpeed * 100)) % 10);
        
        ssd1306_SetCursor(0,0);
        ssd1306_WriteString(buf[0],Font_7x10,White);
        ssd1306_SetCursor(0,12);
        ssd1306_WriteString(buf[1],Font_7x10,White);

        ssd1306_UpdateScreen();
        osDelay(100);
    }
}

void oledTaskLaunch(void) {
    printf("launching task \"%s\"\n",TASK_NAME);
    oledTaskHandle = osThreadNew(task,NULL,&taskAttributes);
}

