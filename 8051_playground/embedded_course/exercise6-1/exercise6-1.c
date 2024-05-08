#include <reg51.h>

#define CODE_L (0x38)
#define CODE_2 (0x5b)
#define CODE_H (0x76)
#define CODE_3 (0x4f)

// 12MHz 16bit 50ms
#define TH0_VAL (0x3c)
#define TL0_VAL (0xb0)

unsigned char timer0_cnt = 0;
unsigned char table[2][2] = {{CODE_L,CODE_2},{CODE_H,CODE_3}};
bit displayH3 = 1;

// 12.000MHz
void delay1ms() {
	unsigned char i, j;

	i = 2;
	j = 239;
	do {
		while (--j);
	} while (--i);
}

// 1. timer + inter
// 2. timer + main loop request

//#define USE_TIMER
#ifdef USE_TIMER

void initTimerAndInter() {
	TMOD = 0x01;
	TH0 = TH0_VAL;
	TL0 = TL0_VAL;
	TR0 = 1;
	ET0 = 1;
	EA = 1;
}

void t0InterServ() interrupt 1 {
	TH0 = TH0_VAL;
	TL0 = TL0_VAL;
	if(++timer0_cnt == 20) {
		timer0_cnt = 0;
		displayH3 = !displayH3;
	}
}

int main() {
	initTimerAndInter();
	while(1) {
		P3 = 0x01;
		P2 = table[displayH3][1];
		delay1ms();
		P3 = 0x02;
		P2 = table[displayH3][0];
		delay1ms();
	}
}

#else

void initTimer() {
	TMOD = 0x01;
	TH0 = TH0_VAL;
	TL0 = TL0_VAL;
	TR0 = 1;
}

int main() {
	initTimer();
	while(1) {
		if(TF0) {
			TF0 = 0;
			TH0 = TH0_VAL;
			TL0 = TL0_VAL;
			if(++timer0_cnt == 20) {
				timer0_cnt = 0;
				displayH3 = !displayH3;
			}
		}
		
		P3 = 0x01;
		P2 = table[displayH3][1];
		delay1ms();
		P3 = 0x02;
		P2 = table[displayH3][0];
		delay1ms();
	}
}

#endif