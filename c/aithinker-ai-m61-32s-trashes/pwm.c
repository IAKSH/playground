#ifndef BL616
#define BL616
#endif

#include "bflb_mtimer.h"       &nbsp;//mtimer定时器头文件
#include "bflb_pwm_v2.h"     &nbsp;//pwm_v2头文件
#include "bflb_clock.h"           //系统时钟头文件
#include "board.h"                 //开发板头文件，包装的库
#include "bflb_gpio.h"           //gpio头文件

struct bflb_device_s *pwm;     //创建LHAL外设库结构体，名称为pwm
  
void my_pwm_gpio_init()       &nbsp;//编写一个选择pwm输出的gpio口初始化函数
{
    struct bflb_device_s *gpio;
    gpio = bflb_device_get_by_name("gpio");

    bflb_gpio_init(gpio, GPIO_PIN_12, GPIO_FUNC_PWM0 | GPIO_ALTERNATE | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_1);
//选择IO0作为pwm输出，
}

int main(void)
{
    int i;                                   //临时变量i，作为改变占空比的变量
    board_init();                       //板子初始化
    my_pwm_gpio_init();         //调用函数，里面设置好了pwm输出的gpio口
  
    pwm = bflb_device_get_by_name("pwm_v2_0"); &nbsp;//给外设接口赋名pwm_v2_0
    /* period = .XCLK / .clk_div / .period = 40MHz / 40 / 1000 = 1KHz */
  
    struct bflb_pwm_v2_config_s cfg = {
        .clk_source = BFLB_SYSTEM_XCLK,
        .clk_div = 40,
        .period = 1000,
    };               &nbsp;//设置PWM的频率，选择时钟，分频，和周期。根据上面的公式算出最终的频率。

    /*初始化PWM输出*/
    bflb_pwm_v2_init(pwm, &cfg);
    bflb_pwm_v2_start(pwm);         &nbsp;//将设置好的频率开启pwm输出

    while (1) {
    //蓝灯呼吸亮灭 &nbsp;
    bflb_pwm_v2_channel_positive_start(pwm, PWM_CH0);         //那么问题来了，如何知道IO口对应的PWM通道，后面会解答，IO0是通道0
    for(i=150;i>0;i--)
    {
        bflb_pwm_v2_channel_set_threshold(pwm, PWM_CH0, i, 150); //改变占空比，变量i会不断变化
        bflb_mtimer_delay_ms(10);
    }
    
    for(i=1;i<150;i++)
    {
        bflb_pwm_v2_channel_set_threshold(pwm, PWM_CH0, i, 150);
        bflb_mtimer_delay_ms(10);
    }
    bflb_pwm_v2_channel_positive_stop(pwm, PWM_CH0);       &nbsp;
    }
}