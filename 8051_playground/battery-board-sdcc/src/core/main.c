#include <stdint.h>
#include <drivers/register/stc8g.h>
#include <drivers/uart/uart.h>
#include <drivers/i2c/i2c.h>
#include <drivers/ssd1306/ssd1306.h>
#include <drivers/ina226/ina226.h>
#include <utils/delay/soft_delay.h>

#define SSD1306_WIDTH     128
#define SSD1306_HEIGHT    32
#define ANIM_WIDTH        10  // 动画横条的宽度
#define IDLE_TIME_MAX 15000

static float volt    = 0;
static float current = 0;
static float power   = 0;
static __bit should_sleep = 0;
static uint16_t time_cnt = 0;

static void sleep(void) {
    ssd1306_display_off();
    PCON |= 0x02;
}

// will be called in external isq
static void wakeup(void) {
    PCON &= ~0x02;
    ssd1306_display_on();
}

// 由于某种神秘因素，isq必须和main放在同一个文件
// 可能和abi中的符号名称有关

static void key_init(void) {
    EX1 = 1;
    IT1 = 1;
}

void key_interrupt(void) __interrupt(2) {
    if (should_sleep) {
        wakeup();
        should_sleep = 0;
        TR0 = 1;
    }
}

// Timer0 1ms @24.000MHz
static void timer_init(void) {
	AUXR |= 0x80;			//定时器时钟1T模式
	TMOD &= 0xF0;			//设置定时器模式
	TL0 = 0x40;				//设置定时初始值
	TH0 = 0xA2;				//设置定时初始值
	TF0 = 0;				//清除TF0标志
	TR0 = 1;				//定时器0开始计时
    ET0 = 1;                //允许定时器0中断
}

void timer_interrupt(void) __interrupt(1) {
    if(++time_cnt >= IDLE_TIME_MAX) {
        should_sleep = 1;
        time_cnt = 0;
        TR0 = 0;
    }
}

static void init_ports(void) {
    /* 配置端口 */
    P3M0 = 0x00;
    P3M1 = 0x00;
    P5M0 = 0x00;
    P5M1 = 0x00;
    // 设置 P3.2 为 INA226 的 Alert 输入
    P3M0 |= 0x04;
    P3M1 |= 0x04;
}

static void init_interrupts(void) {
    EA  = 1;
}

int main(void) {
    init_ports();
    init_interrupts();
    uart_init();

    uart_send_string("hello world!\n");
    i2c_init();
    uart_send_string("i2c inited\n");
    ina226_init();
    uart_send_string("ina226 inited\n");
    ssd1306_init();
    uart_send_string("ssd1306 inited\n");

    ssd1306_clear();
    ssd1306_draw_str(0, 0, (uint8_t*)"Hello wrold!", 16);
    soft_delay_100ms();
    ssd1306_draw_str(0, 2, (uint8_t*)"STC8G @24MHz", 8);
    ssd1306_draw_str(0, 3, (uint8_t*)"BatteryBoard 2.0", 8);
    for(int i = 0;i < 1;i++)
        soft_delay_1sec();
    ssd1306_clear();

    timer_init();
    key_init();

    /* 绘制静态UI：测量值文字信息（固定区域） */
    ssd1306_draw_str(0, 0, (uint8_t*)"V:", 8);      // 电压信息，在页0（y=0~7）
    ssd1306_draw_str(70, 0, (uint8_t*)"A:", 8);     // 电流信息
    ssd1306_draw_str(0, 2, (uint8_t*)"W:", 8);      // 功率信息，在页2（y=16~23）
    ssd1306_draw_str(70, 2, (uint8_t*)"L:", 8);     // 负载百分比

    uart_send_string("UI initialized\n");

    // 动画区域参数：动态横条在动画区域内移动，位置记录变量
    uint8_t dynamic_x = 0;          // 动画横条左侧的x坐标
    int8_t dynamic_dir = 1;         // 移动方向，1表示向右，-1表示向左
    const uint8_t dynamic_y = 30;   // 动画横条所在的y坐标

    while (1) {
        if(should_sleep)
            sleep();

        // 更新传感器数据
        volt    = ina226_read_volt();
        current = ina226_read_current();
        power   = ina226_read_power();

        /* 每次循环更新测量值显示
           （注意：此处简单调用绘制函数，假设新数值完全覆盖旧值） */
        ssd1306_draw_float(20, 0, volt, 2, 8);
        ssd1306_draw_float(90, 0, current, 2, 8);
        ssd1306_draw_float(20, 2, power, 2, 8);
        ssd1306_draw_float(90, 2, (volt / 12.0f) * 100.0f, 2, 8);

        /* 更新动画区域 -- 仅清除上一次动画横条的像素 */
        for (uint8_t i = 0; i < ANIM_WIDTH; i++) {
            ssd1306_clear_point(dynamic_x + i, dynamic_y);
        }
        // 更新动画坐标：确保横条在 0～(SSD1306_WIDTH - ANIM_WIDTH) 之间来回移动
        if (dynamic_x == 0) {
            dynamic_dir = 1;
        } else if (dynamic_x >= (SSD1306_WIDTH - ANIM_WIDTH)) {
            dynamic_dir = -1;
        }
        dynamic_x += dynamic_dir;
        // 在新的位置绘制动画横条
        for (uint8_t i = 0; i < ANIM_WIDTH; i++) {
            ssd1306_draw_point(dynamic_x + i, dynamic_y);
        }

        // 输出调试信息：UART输出缩放后的电压值
        uart_send_uint16((int)(volt * 10.0f));

        soft_delay_10ms();
    }
}
