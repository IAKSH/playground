// 似乎可以扩充到无限路
// 不过随着路数增加，pwm脉冲会越来越不精确，甚至可能彻底崩坏
// 而且，各路的PWM波形并不是同步生成的

#include <mcs51/8051.h>
#include <stdbool.h>

unsigned int cycle[4] = {0,0,0,0};
unsigned int duty[4] = {0,0,0,0};
bool increase = true;

#define _nop_() __asm NOP __endasm
#define PWM_PIN P1_0
#define ON 0
#define OFF 1

//100us @11.0592MHz
void init_timer0(void)
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

void increase_cycle_1(void) __interrupt(1)
{
    TR0 = 0;
	TL0 = 0xAE;	
	TH0 = 0xFB;
    TR0 = 1;

    P1_0 = (++cycle[0] > duty[0]);
    if(cycle[0] == 100) cycle[0] = 0;

    P1_1 = (++cycle[1] > duty[1]);
    if(cycle[1] == 100) cycle[1] = 0;
    
    P1_2 = (++cycle[2] > duty[2]);
    if(cycle[2] == 100) cycle[2] = 0;

    P1_3 = (++cycle[3] > duty[3]);
    if(cycle[3] == 100) cycle[3] = 0;
}

//@11.0592MHz
void delay_1ms()
{
	unsigned char i, j;

	_nop_();
	i = 11;
	j = 190;
	do
	{
		while (--j);
	} while (--i);
}

void main(void)
{
    init_timer0();

    //duty[0] = 1;
    //duty[1] = 5;
    //P1 = 0x00;
    while(true)
    {
        // pwm 0
        if(duty[0] == 100) increase = false;
        else if(duty[0] == 0) increase = true;

        if(increase) ++duty[0];
        else --duty[0];

        // pwm 1
        if(duty[1] == 100) increase = false;
        else if(duty[1] == 0) increase = true;

        if(increase) ++duty[1];
        else --duty[1];

        // pwm 2
        if(duty[2] == 100) increase = false;
        else if(duty[2] == 0) increase = true;

        if(increase) ++duty[2];
        else --duty[2];

        // pwm 3
        if(duty[3] == 100) increase = false;
        else if(duty[3] == 0) increase = true;

        if(increase) ++duty[3];
        else --duty[3];

        delay_1ms();
    }
}