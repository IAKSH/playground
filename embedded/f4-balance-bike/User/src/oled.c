#include "main.h"
#include "driver/ssd1306.h"
#include "driver/ssd1306_tests.h"
#include "driver/ssd1306_fonts.h"
#include "driver/gyro.h"
#include <stdio.h>
#include <math.h>

extern float mpu6050_pitch,mpu_6050_roll,mpu_6050_yaw;
extern double pitchPIDOutput;

void oled_task(void* arg) {
    ssd1306_Init();
    printf("ssd1306 initialized!\n");
    ssd1306_Fill(Black);

    ssd1306_SetCursor(0,0);
    ssd1306_WriteString("init MPU6050...",Font_7x10,White);
    ssd1306_UpdateScreen();
    
    osEventFlagsWait(event,EVENT_FLAG_GYRO_INITIALIZED,osFlagsWaitAny,osWaitForever);
    ssd1306_Fill(Black);

    char buf[2][32];

    while(1) {
        snprintf(buf[0],sizeof(buf),"p= %d.%d%d r= %d.%d%d      ",
        (int)mpu6050_pitch,(int)fabs((int)(mpu6050_pitch * 10)) % 10,(int)fabs((int)(mpu6050_pitch * 100)) % 10,
        (int)mpu_6050_roll,(int)fabs((int)(mpu_6050_roll) * 10) % 10,(int)fabs((int)(mpu_6050_roll * 100)) % 10);

        ssd1306_SetCursor(0,0);
        ssd1306_WriteString(buf[0],Font_7x10,White);
        ssd1306_SetCursor(0,12);
        ssd1306_WriteString(buf[1],Font_7x10,White);

        ssd1306_UpdateScreen();
        osDelay(100);
    }
}
