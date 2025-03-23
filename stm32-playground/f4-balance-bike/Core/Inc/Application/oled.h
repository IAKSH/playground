#pragma once
#include "main.h"
#include "cmsis_os2.h"

extern osThreadId_t oledTaskHandle;
void oledTaskLaunch(void);