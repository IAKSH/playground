#include <mcs51/8051.h>
#include <stdbool.h>
#include "pwm_motor.h"
#include "front_detector.h"
#include "collision_detector.h"
#include "line_detector.h"
#include "side_detector.h"
#include "soft_delay.h"

void move(void)
{
    left_pwm.duty = 30;
    right_pwm.duty = 30;

    if(LD_check_1())
        right_pwm.duty += 35;
    if(LD_check_2())
        right_pwm.duty += 35;
    if(LD_check_3())
        left_pwm.duty += 35;
    if(LD_check_4())
        left_pwm.duty += 35;

    //if(left_pwm.duty > 100)
    //    left_pwm.duty = 100;
    //if(right_pwm.duty > 100)
    //    right_pwm.duty = 100;
}

void check_collision(void)
{
    // 发生撞击
    if(CD_check())
    {
        // 后退一段距离，以留出调头空间
        left_pwm.forward = false;
        right_pwm.forward = false;
        left_pwm.duty = 100;
        right_pwm.duty = 100;
        for(int i = 0;i < 12;i++)
            delay_1ms();

        // 调头
        left_pwm.forward = true;
        for(int i = 0;i < 7;i++)
            delay_1ms();
        while(!(LD_check_2() && LD_check_3()));
 
        right_pwm.forward = true;
        left_pwm.duty = 30;
        right_pwm.duty = 30;
    }
}

void turn_right(void)
{
    left_pwm.duty = 100;
    right_pwm.duty = 100;
    right_pwm.forward = false;
    
    while(!(LD_check_2() && LD_check_3()));

    right_pwm.forward = true;
}

void wait_and_turn_left(void)
{
    for(int i = 0;i < 5;i++)
        delay_1ms();

    right_pwm.duty = 100;
    left_pwm.duty = 100;
    left_pwm.forward = false;

    for(int i = 0;i < 10;i++)
        delay_1ms();
    while(!(LD_check_2() && LD_check_3()));

    left_pwm.forward = true;
}

void check_turning(void)
{
    // 能左转就（准备）左转
    if(SD_check_ever_left())
    {
        wait_and_turn_left();
        SD_reload();
    }
    // 不能左转且出线，做进一步判断
    else if(!(LD_check_1() || LD_check_2() || LD_check_3() || LD_check_4()))
    {
        if(SD_check_ever_right())
            turn_right();
        SD_reload();
    }
}

//void check_crossing(void)
//{
//    // 实际上是实现了一个迷宫算法
//    // 当全部LD检测到线时，考虑十字路口
//    // 十字路口的特征是，全部LD和两侧SD都检测到路线
//    if(LD_check_1() && LD_check_2() && LD_check_3() && LD_check_4() && SD_check_left() && SD_check_right())
//    {
//        // 由于是自线内开始的转向，需要为之前的实现延迟检测
//        left_pwm.duty = 100;
//        right_pwm.forward = false;
//        right_pwm.duty = 100;
//
//        for(int i = 0;i < 25;i++)
//            delay_1ms();
//        while(!LD_check_4());
//        while(!LD_check_1());
//
//        right_pwm.forward = true;
//    }
//}

void main(void)
{
    PWM_init();
    CD_init();
    SD_init();

    //// 等待移除挡板
    left_pwm.duty = 0;
    right_pwm.duty = 0;
    while(FD_check());

    while(true)
    {
        // 前进/后退并修正偏移
        move();
        // 处理可能的碰撞
        check_collision();
        // 处理可能的拐弯
        check_turning();
    }
} 