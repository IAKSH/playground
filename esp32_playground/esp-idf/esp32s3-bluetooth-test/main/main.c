#include "task_ble_gatt.h"
#include "task_posture.h"
#include "task_pwm_test.h"

#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void) {
    xTaskCreate(ble_gatts_main,"ble_gatts",8192,NULL,1,NULL);
    xTaskCreate(posture_main,"posture",8192,NULL,1,NULL);
    xTaskCreate(pwm_test_main,"pwm_test",8192,NULL,1,NULL);

    //xTaskCreate(TaskFun,TaskName,StackSize,Param,Priority,*Task)
    //1:TaskFun 任务函数
    //2:TaskName 任务名字
    //3:StackSize 任务堆栈大小
    //4:Param 任务传入参数
    //5:Priority 任务优先级,最低优先级为0=空闲任务,可以设置0-31
    //6:Task 任务句柄任务创建成功后会返回这个句柄,其他api任务接口可调用这个句柄
}