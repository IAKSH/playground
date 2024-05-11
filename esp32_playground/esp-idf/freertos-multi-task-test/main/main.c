#include <stdio.h>
#include <stdbool.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <sdkconfig.h>
#include <driver/gpio.h>
#include <esp_log.h>

#define BOARD_LED_PIN (2)

#define SEG_7_DATA_1 (4)
#define SEG_7_DATA_2 (5)
#define SEG_7_DATA_3 (18)

#define SEG_7_CHANNEL_1 (27)
#define SEG_7_CHANNEL_2 (26)
#define SEG_7_CHANNEL_3 (22)
#define SEG_7_CHANNEL_4 (23)

uint8_t d = 0;

void init_board_led_pin(void) {
    gpio_reset_pin(BOARD_LED_PIN);
    gpio_set_direction(BOARD_LED_PIN,GPIO_MODE_OUTPUT);
}

void board_led_blink_task(void* param) {
    init_board_led_pin();
    bool board_led_state = false;
    while(1) {
        ++d;
        gpio_set_level(BOARD_LED_PIN,board_led_state);
        vTaskDelay((board_led_state ? 10 : 990) /portTICK_PERIOD_MS);
        board_led_state = !board_led_state;
    }
}

void init_seg_7_data_pin(void) {
    gpio_reset_pin(SEG_7_DATA_1);
    gpio_set_direction(SEG_7_DATA_1,GPIO_MODE_OUTPUT);
    
    gpio_reset_pin(SEG_7_DATA_2);
    gpio_set_direction(SEG_7_DATA_2,GPIO_MODE_OUTPUT);

    gpio_reset_pin(SEG_7_DATA_3);
    gpio_set_direction(SEG_7_DATA_3,GPIO_MODE_OUTPUT);
}

void init_seg_7_channel_pin(void) {
    gpio_reset_pin(SEG_7_CHANNEL_1);
    gpio_set_direction(SEG_7_CHANNEL_1,GPIO_MODE_OUTPUT);
    
    gpio_reset_pin(SEG_7_CHANNEL_2);
    gpio_set_direction(SEG_7_CHANNEL_2,GPIO_MODE_OUTPUT);

    gpio_reset_pin(SEG_7_CHANNEL_3);
    gpio_set_direction(SEG_7_CHANNEL_3,GPIO_MODE_OUTPUT);

    gpio_reset_pin(SEG_7_CHANNEL_4);
    gpio_set_direction(SEG_7_CHANNEL_4,GPIO_MODE_OUTPUT);
}

#define SEG_7_A {0,0,0}
#define SEG_7_B {0,1,1}
#define SEG_7_C {1,0,1}
#define SEG_7_D {1,1,0}
#define SEG_7_E {0,1,0}
#define SEG_7_F {1,0,0}
#define SEG_7_DOT {0,0,1}
#define SEG_7_NONE {1,1,1}

void seg_7_update_data_task(void* param) {
    const char* TAG = "seg_7_update_data_task";
    const char PINS[4] = {SEG_7_CHANNEL_1,SEG_7_CHANNEL_2,SEG_7_CHANNEL_3,SEG_7_CHANNEL_4};
    bool state_138[3] = SEG_7_NONE;
    // 000 -> a
    // 001 -> dot
    // 010 -> e
    // 011 -> b
    // 100 -> f
    // 101 -> c
    // 110 -> d
    // 111 -> none

    /*
    0: a + b + c + d + e + f
    1: e + c
    2: a + b + d + e + g
    3: a + b + c + d + g
    4: b + c + f + g
    5: a + c + d + f + g
    6: a + c + d + e + f + g
    7: a + b + c
    8: a + b + c + d + e + f + g
    9: a + b + c + d + f + g
    A: a + b + c + e + f + g
    B: c + d + e + f + g
    C: a + d + e + f
    D: b + c + d + e + g
    E: a + d + e + f + g
    F: a + e + f + g
    */

    init_seg_7_data_pin();
    init_seg_7_channel_pin();

    uint8_t i,j,k;
    
    uint8_t code[6][3] = {
        SEG_7_A,
        SEG_7_B,
        SEG_7_C,
        SEG_7_D,
        SEG_7_E,
        SEG_7_F
    };

    while(1) {
        for(i = 0;i < 4;i++) {
            // change channel
            for(j = 0;j < 4;j++)
                gpio_set_level(PINS[j], 0);
            gpio_set_level(PINS[i], 1);
            // send data
            for(j = 0;j < 3;j++) {
                for(k = 0;k < 3;k++)
                    state_138[k] = code[(i + j + d) % 6][k];
                gpio_set_level(SEG_7_DATA_1, state_138[0]);
                gpio_set_level(SEG_7_DATA_2, state_138[1]);
                gpio_set_level(SEG_7_DATA_3, state_138[2]);
                vTaskDelay(1 / portTICK_PERIOD_MS);
            }
            vTaskDelay(10 / portTICK_PERIOD_MS);
            //ESP_LOGI(TAG, "[%d,%d,%d]",state_138[0],state_138[1],state_138[2]);            
        }
    }
}

void app_main(void) {
    // 很多教程的堆栈大小用的1024，太小了，很容易炸，需要换大一点的
    xTaskCreate(board_led_blink_task,"board_led_blink_task",8192,NULL,1,NULL);
    xTaskCreate(seg_7_update_data_task,"seg_7_update_task",8192,NULL,1,NULL);

    //xTaskCreate(TaskFun,TaskName,StackSize,Param,Priority,*Task)
    //1:TaskFun 任务函数
    //2:TaskName 任务名字
    //3:StackSize 任务堆栈大小
    //4:Param 任务传入参数
    //5:Priority 任务优先级,最低优先级为0=空闲任务,可以设置0-31
    //6:Task 任务句柄任务创建成功后会返回这个句柄,其他api任务接口可调用这个句柄    
}