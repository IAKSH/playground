#pragma once

typedef struct {
    float euler[3],accel[3],temperature;
} Mpu6050Results;

extern Mpu6050Results mpu6050_results;

void mpu6050_main(void);