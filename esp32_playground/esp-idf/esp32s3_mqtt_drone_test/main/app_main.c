//#include "mqtt_trans.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "task/tasks.h"
#include "wireless/mqtt_trans.h"

void app_main(void)
{
    // TODO: 自检

    //xTaskCreate(task_power,"power",4096,NULL,5,NULL);
    xTaskCreate(task_posture,"posture",4096,NULL,5,NULL);
    xTaskCreate(task_stabilizer,"stabilizer",4096,NULL,5,NULL);
    //xTaskCreate(task_camera,"camera",4096,NULL,5,NULL);
    //xTaskCreate(task_led,"led",4096,NULL,5,NULL);

    mqtt_startup();
}
