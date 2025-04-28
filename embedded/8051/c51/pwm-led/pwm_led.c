#include <reg51.h>

sbit LED1 = P1^0;
sbit LED2 = P1^1;
sbit LED3 = P1^2;
sbit LED4 = P1^3;

unsigned char pwm[4] = {16,32,64,128};
unsigned char dir[4] = {0,0,0,0};

void timer0_init() {
    TMOD = 0x01;
    TH0 = 0xFF;
    TL0 = 0xFF;
    ET0 = 1;
    EA = 1;
    TR0 = 1;
}

void timer0_isr() interrupt 1 {
    static unsigned char pwm_count = 0;
    TH0 = 0xFF;
    TL0 = 0xFF;
    pwm_count++;
    LED1 = (pwm_count < pwm[0]) ? 0 : 1;
    LED2 = (pwm_count < pwm[1]) ? 0 : 1;
    LED3 = (pwm_count < pwm[2]) ? 0 : 1;
    LED4 = (pwm_count < pwm[3]) ? 0 : 1;
}

//@11.0592MHz
void delay10us() {
	unsigned char i;
	i = 2;
	while (--i);
}

void update_pwm() {
	unsigned char i;
	for(i = 0;i < 4;i++) {
		if(pwm[i] == 0 || pwm[i] == 255)
			dir[i] = !dir[i];
		pwm[i] += (dir[i] ? 1 : -1);
	}
}

void main() {
    timer0_init();
    while(1) {
        update_pwm();
		delay10us();
    }
}
