#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "stm32f1xx.h"

void tim_irq(TIM_HandleTypeDef *htim);

#ifdef __cplusplus
}
#endif