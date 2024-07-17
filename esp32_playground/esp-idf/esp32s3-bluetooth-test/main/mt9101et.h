#pragma once

#include "kalman.h"

#include <stdint.h>

void mt9101et_init(void);
void mt9101et_read(uint32_t* raw,uint32_t* voltage);

typedef struct {
    KalmanState raw,volt;
} Mt9101etKalmanState;

void mt9101et_kalman_init(Mt9101etKalmanState* kalman_state);
void mt9101et_kalman_update(Mt9101etKalmanState* kalman_state, uint32_t* raw,uint32_t* volt);