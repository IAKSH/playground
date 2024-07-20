#pragma once

typedef enum {
    LED_STATUS_PWM,     // Normal
    LED_STATUS_BLINK,   // Initializing
    LED_STATUS_ON       // Error
} led_status_t;

void led_apply_status(led_status_t status);