#include "board.h"
#include <FreeRTOS.h>
#include <task.h>
#include <stdbool.h>
#include <stdio.h>
#include <bl616_glb.h>
#include <bl616_gpio.h>
#include <bl616_glb_gpio.h>

#define DBG_TAG "MAIN"

#define PIN_LED_R GPIO_PIN_12
#define PIN_LED_G GPIO_PIN_14
#define PIN_LED_B GPIO_PIN_15
#define PIN_LED_4 GPIO_PIN_29
#define PIN_LED_5 GPIO_PIN_27

int main(void) {
    board_init();

    struct bflb_device_s* gpio = bflb_device_get_by_name("gpio");
    bflb_gpio_init(gpio,PIN_LED_R,GPIO_OUTPUT | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_0);
    bflb_gpio_init(gpio,PIN_LED_G,GPIO_OUTPUT | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_0);
    bflb_gpio_init(gpio,PIN_LED_B,GPIO_OUTPUT | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_0);
    bflb_gpio_init(gpio,PIN_LED_4,GPIO_OUTPUT | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_0);
    bflb_gpio_init(gpio,PIN_LED_5,GPIO_OUTPUT | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_0);

    bflb_gpio_set(gpio,PIN_LED_R);
    bflb_gpio_set(gpio,PIN_LED_G);
    bflb_gpio_set(gpio,PIN_LED_B);
    bflb_gpio_set(gpio,PIN_LED_4);
    bflb_gpio_set(gpio,PIN_LED_5);

	char led_flag = 0;

    while(1) {
        if(led_flag) {
            bflb_gpio_set(gpio,PIN_LED_4);
            bflb_gpio_reset(gpio,PIN_LED_5);
            led_flag = 0;
        }
        else {
            bflb_gpio_reset(gpio,PIN_LED_4);
            bflb_gpio_set(gpio,PIN_LED_5);
            led_flag = 1;
        }
        bflb_mtimer_delay_ms(500);
    }
}