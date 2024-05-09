#include "board.h"
#include <FreeRTOS.h>
#include <task.h>
#include <stdbool.h>
#include <stdio.h>
#include <bl616_glb.h>
#include <bl616_gpio.h>
#include <bl616_glb_gpio.h>

#define DBG_TAG "MAIN"

#define PIN_LED_R (12)
#define PIN_LED_G (14)
#define PIN_LED_B (15)
#define PIN_LED_4 (29)
#define PIN_LED_5 (27)

void init_gpio_as_output(unsigned int gpio)
{
    GLB_GPIO_Cfg_Type cfg;
    cfg.drive = 0;
    cfg.smtCtrl = 1;
    cfg.gpioFun = GPIO_FUN_GPIO;
    cfg.outputMode = 0;
    cfg.pullType = GPIO_PULL_NONE;

    cfg.gpioPin = gpio;
    cfg.gpioMode = GPIO_MODE_OUTPUT;
    GLB_GPIO_Init(&cfg);
}

int main(void)
{
    board_init();
    
    init_gpio_as_output(PIN_LED_R);
    init_gpio_as_output(PIN_LED_G);
    init_gpio_as_output(PIN_LED_B);
    init_gpio_as_output(PIN_LED_4);
    init_gpio_as_output(PIN_LED_5);

    GLB_GPIO_Write(PIN_LED_R, 1);
    GLB_GPIO_Write(PIN_LED_G, 1);
    GLB_GPIO_Write(PIN_LED_B, 1);
    GLB_GPIO_Write(PIN_LED_4, 1);
    GLB_GPIO_Write(PIN_LED_5, 1);

    while(1);
}