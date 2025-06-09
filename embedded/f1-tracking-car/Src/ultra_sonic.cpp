#include "ultra_sonic.hpp"
#include <cstdio>
#include <cinttypes>

car::UlatraSonic::UlatraSonic(Ticker& ticker,osMutexId_t* mutex,GPIO_TypeDef* trig_port,uint16_t trig_pin)
    : ticker(ticker),mutex(mutex),trig_port(trig_port),trig_pin(trig_pin) {}

car::UlatraSonic::UlatraSonic(Ticker& ticker,GPIO_TypeDef* trig_port,uint16_t trig_pin)
    : ticker(ticker),trig_port(trig_port),trig_pin(trig_pin) {}

void car::UlatraSonic::trig() {
    HAL_GPIO_WritePin(trig_port,trig_pin,GPIO_PIN_SET);
    //tick_start = ticker.get_us();
    //while(ticker.get_us() - tick_start < 15);
    HAL_Delay(1);
    HAL_GPIO_WritePin(trig_port,trig_pin,GPIO_PIN_RESET);
    data_ready = false;
}

void car::UlatraSonic::irq_rising() {
    ticker.clear();
    tick_start = ticker.get_us();
}

void car::UlatraSonic::irq_falling() {
    tick_end = ticker.get_us();
    uint32_t diff = tick_end - tick_start;
    if(diff > 0)
        distance = (diff * 1.0f) * 0.017f; // 0.017 = (340 m/s / 2) / 1000 (ms to s, cm)
    data_ready = true;
}

float car::UlatraSonic::UlatraSonic::get_distance() {
    //printf("[debug] start: %d\tend: %d\tcnt: %d\n",tick_start,tick_end,ticker.get_cnt());
    //printf("[debug] duration_us: %d\n",tick_end - tick_start);
    // maybe we don't need mutex here
    float ret = 0;
    if(mutex) {
        osMutexAcquire(*mutex,osWaitForever);
        ret = distance;
        osMutexRelease(*mutex);
    }
    else
        ret = distance;
    return ret;
}

bool car::UlatraSonic::ready() {
    return data_ready;
}
