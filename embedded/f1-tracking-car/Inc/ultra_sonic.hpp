#pragma once
#include "main.h"
#include "ticker.hpp"

namespace car {
    class UlatraSonic {
    private:
        Ticker& ticker;
        osMutexId_t* mutex;
        GPIO_TypeDef* trig_port;
        uint16_t trig_pin;
        uint32_t tick_start,tick_end;
        bool data_ready;
        float distance;
        void caculate_distance();

    public:
        UlatraSonic(Ticker& ticker,osMutexId_t* mutex,GPIO_TypeDef* trig_port,uint16_t trig_pin);
        UlatraSonic(Ticker& ticker,GPIO_TypeDef* trig_port,uint16_t trig_pin);
        void trig();
        void irq_rising();
        void irq_falling();
        float get_distance();
        bool ready();
    };
}