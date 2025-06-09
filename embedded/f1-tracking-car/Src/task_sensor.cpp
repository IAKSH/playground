#include "tasks.h"
#include "main.h"
#include "status.hpp"
#include <cstdio>

car::Ticker ticker(&htim5);
car::UlatraSonic ultra_sonic(ticker,ULTRA_SONIC_TRIG_GPIO_Port,ULTRA_SONIC_TRIG_Pin);

float distance;

void sensor_task(void* args) {
    ticker.start();
    while(true) {
        osSemaphoreAcquire(sensor_timer_sem,osWaitForever);
        ultra_sonic.trig();
        //while(!ultra_sonic.ready());
        HAL_Delay(10);

        distance = ultra_sonic.get_distance();
        //printf("[debug] ultra_sonic: %d.%d%d\n",
        //    static_cast<int>(distance),
        //    static_cast<int>(distance * 10) % 10,
        //    static_cast<int>(distance * 100) % 10
        //);
    }
}