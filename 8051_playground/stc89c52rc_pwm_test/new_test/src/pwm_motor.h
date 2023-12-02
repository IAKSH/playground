#pragma once

#include <mcs51/8051.h>

#define LEFT_PWM_FORWARD_PIN  P1_0  //int1
#define LEFT_PWM_BACKWARD_PIN P1_1  //int2
#define RIGHT_PWM_FORWARD_PIN P1_3 //int4
#define RIGHT_PWM_BACKWARD_PIN P1_2  //int3

//100us @11.0592MHz
inline static void PWM_init_timer0(void)
{
	PCON |= 0x80;
	TMOD &= 0xF0;
	TL0 = 0xAE;	
	TH0 = 0xFB;	
	TF0 = 0;	
	EA = 1;
	ET0 = 1;
	TR0 = 1;	
}

typedef struct
{
    unsigned int cycle;// 周期计数
    unsigned int duty;// 占空比 [0,100] (%)   
    bool forward;// 是否正转
    
}
PWM_Motor;

inline static PWM_Motor left_pwm,right_pwm;

inline static void PWM_init()
{
    PWM_init_timer0();
    left_pwm.cycle = 0;
    left_pwm.duty = 0;
    left_pwm.forward = true;
    right_pwm.cycle = 0;
    right_pwm.duty = 0;
    right_pwm.forward = true;

    P1 = 0x00;
}

//inline static void PWM_interrupt(void) __interrupt 1
inline static void PWM_interrupt(void) __interrupt(1)
{
    TR0 = 0;
	TL0 = 0xAE;
	TH0 = 0xFB;
    TR0 = 1;

    if(left_pwm.forward)
    {
        LEFT_PWM_BACKWARD_PIN = 0;
        LEFT_PWM_FORWARD_PIN = !(++left_pwm.cycle > left_pwm.duty);
    }
    else
    {
        LEFT_PWM_FORWARD_PIN = 0;
        LEFT_PWM_BACKWARD_PIN = !(++left_pwm.cycle > left_pwm.duty);
    }
    if(left_pwm.cycle == 100) left_pwm.cycle = 0;

    if(right_pwm.forward)
    {
        RIGHT_PWM_BACKWARD_PIN = 0;
        RIGHT_PWM_FORWARD_PIN = !(++right_pwm.cycle > right_pwm.duty);
    }
    else
    {
        RIGHT_PWM_FORWARD_PIN = 0;
        RIGHT_PWM_BACKWARD_PIN = !(++right_pwm.cycle > right_pwm.duty);
    }
    if(right_pwm.cycle == 100) right_pwm.cycle = 0;
}
