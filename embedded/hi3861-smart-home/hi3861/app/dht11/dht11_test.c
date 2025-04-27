#include <stdio.h>
#include <unistd.h>
#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_gpio.h"
#include "hi_io.h"
#include "hi_time.h"

#define DHT11_TIMEOUT 100 // 超时时间，单位ms

static void dht11_begin_output(void) {
    IoTGpioSetDir(HI_IO_NAME_GPIO_10, IOT_GPIO_DIR_OUT);
}

static void dht11_begin_input(void) {
    IoTGpioSetDir(HI_IO_NAME_GPIO_10, IOT_GPIO_DIR_IN);
    //hi_io_set_pull(HI_IO_NAME_GPIO_10, HI_IO_PULL_NONE);
}

static void dht11_pin_set(int i) {
    IoTGpioSetOutputVal(HI_IO_NAME_GPIO_10, i);
}

static int dht11_pin_read(void) {
    IotGpioValue val;
    IoTGpioGetInputVal(HI_IO_NAME_GPIO_10, &val);
    return val;
}

static void dht11_start(void) {
    dht11_begin_output();
    dht11_pin_set(0);
    hi_udelay(20000);  // 拉低至少 18ms
    dht11_pin_set(1);
    hi_udelay(40);  // 拉高 20-40us
    dht11_begin_input();
}

static char dht11_read_byte(void) {
    unsigned char i = 0;
    unsigned char data = 0;
    for (; i < 8; i++) {
        unsigned int timeout = DHT11_TIMEOUT;
        while (dht11_pin_read() == 0 && --timeout > 0); // 加入超时检测
        if (timeout == 0) {
            printf("[dht11] Timeout while reading byte\n");
            return -1;
        }
        hi_udelay(50); // 调整延时，确保传感器有足够时间
        data <<= 1;
        if (dht11_pin_read() == 1) {
            data |= 1;
        }
        timeout = DHT11_TIMEOUT;
        while (dht11_pin_read() == 1 && --timeout > 0); // 加入超时检测
        if (timeout == 0) {
            printf("[dht11] Timeout while reading byte\n");
            return -1;
        }
    }
    return data;
}

osMutexId_t dht11_mutex_id;
unsigned int dht11_data[4];

static void dht11_update_data(void) {
    unsigned int R_H = 0, R_L = 0, T_H = 0, T_L = 0;
    unsigned char RH, RL, TH, TL, CHECK;

    dht11_start();

    if (dht11_pin_read() == 0) { // 判断DHT11是否响应
        //printf("[dht11] Sensor responded\n");
        unsigned int timeout = DHT11_TIMEOUT;
        while (dht11_pin_read() == 0 && --timeout > 0); // 低电平变高电平，等待低电平结束
        if (timeout == 0) {
            printf("[dht11] Timeout waiting for low level end\n");
            return;
        }
        timeout = DHT11_TIMEOUT;
        while (dht11_pin_read() == 1 && --timeout > 0); // 高电平变低电平，等待高电平结束
        if (timeout == 0) {
            printf("[dht11] Timeout waiting for high level end\n");
            return;
        }

        R_H = dht11_read_byte();
        R_L = dht11_read_byte();
        T_H = dht11_read_byte();
        T_L = dht11_read_byte();
        CHECK = dht11_read_byte(); // 接收5个数据

        if (R_H + R_L + T_H + T_L == CHECK) { // 和检验位对比，判断校验接收到的数据是否正确
            RH = R_H;
            RL = R_L;
            TH = T_H;
            TL = T_L;
        } else {
            printf("[dht11] Check failed! RH: %d, RL: %d, TH: %d, TL: %d, CHECK: %d\n", R_H, R_L, T_H, T_L, CHECK);
            return;
        }
    } else {
        printf("[dht11] No response from sensor\n");
    }

    osMutexAcquire(dht11_mutex_id, osWaitForever);
    dht11_data[0] = RH % 100;
    dht11_data[1] = RL % 100;
    dht11_data[2] = TH % 100;
    dht11_data[3] = TL % 100;
    osMutexRelease(dht11_mutex_id);
}

static void dht11_task(void* arg) {
    (void)arg;

    printf("[dht11] startup!\n");

    IoTGpioInit(HI_IO_NAME_GPIO_10);
    hi_io_set_func(HI_IO_NAME_GPIO_10, HI_IO_FUNC_GPIO_10_GPIO);
    hi_io_set_pull(HI_IO_NAME_GPIO_10, HI_IO_PULL_UP);

    while (1) {
        dht11_update_data();
        //printf("[dht11] temp: %d.%d, humi: %d.%d\n", dht11_data[2], dht11_data[3], dht11_data[0], dht11_data[1]);
        osDelay(100);
        osThreadYield(); // 增加任务切换机会
    }
}

static void dht11_entry(void) {
    dht11_mutex_id = osMutexNew(NULL);

    osThreadAttr_t attr = {0};
    attr.name = "dht11_test";
    attr.stack_size = 4096;
    attr.priority = osPriorityNormal1;
    if (osThreadNew((osThreadFunc_t)dht11_task, NULL, &attr) == NULL) {
        printf("[dht11] Failed to create dht11_task!\n");
    }
}

SYS_RUN(dht11_entry);
