#pragma once
#include "main.h"
#include "cmsis_os2.h"

extern osThreadId_t balanceTaskHandle;
void balanceTaskLaunch(void);