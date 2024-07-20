#include "kalman.h"

void kalman_init(KalmanState* state, float q, float r, float p, float initial_value) {
    state->q = q;
    state->r = r;
    state->p = p;
    state->x = initial_value;
}

float kalman_update(KalmanState* state, float measurement) {
    // 预测
    state->p = state->p + state->q;

    // 更新
    state->k = state->p / (state->p + state->r);
    state->x = state->x + state->k * (measurement - state->x);
    state->p = (1 - state->k) * state->p;

    return state->x;
}