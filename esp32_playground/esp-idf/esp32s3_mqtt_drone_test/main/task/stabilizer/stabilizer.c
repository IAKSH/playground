#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "../tasks.h"
#include "motor.h"

void task_stabilizer(void) {
    motor_ledc_initialize();
    motor_ledc_test(3);
    vTaskDelete(NULL);
}
