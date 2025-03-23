#pragma once
#include "main.h"
#include "cmsis_os2.h"

extern osThreadId_t testLEDTaskHandle;
void testLEDTaskLaunch(void);