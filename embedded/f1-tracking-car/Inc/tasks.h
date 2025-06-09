#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void drive_task(void* args);
void led_task(void* args);
void control_task(void* args);
void sensor_task(void* args);

#ifdef __cplusplus
}
#endif