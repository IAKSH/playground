#include <reg51.h>

unsigned char lable[10] = {
	0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};
unsigned char cnt = 0;
unsigned char timer_cycle = 0;

void initT0AndInter() {
	TMOD = 0x01;
	TH0 = 0x3c;
	TL0 = 0xb0;
	TR0 = 0;
	
	EX0 = 1;
	IT0 = 1;
	ET0 = 1;
	EA = 1;
}

void t0InterServ() interrupt 1 {
	TH0 = 0x3c;
	TL0 = 0xb0;
	if(++timer_cycle == 20) {
		timer_cycle = 0;
		cnt = ++cnt % 100;
	}
}

void exInter0Serv() interrupt 0 {
	TR0 = !TR0;
}

// 12MHz
void delay100ms() {
	unsigned char i, j;

	i = 98;
	j = 67;
	do {
		while (--j);
	} while (--i);
}

sbit P3_0 = P3^0;
sbit P3_1 = P3^1;

int main() {
	initT0AndInter();
	P3_0 = 1;
	P3_1 = 0;
	while(1) {
		P3_0 = !P3_0;
		P3_1 = !P3_1;
		P2 = lable[cnt / 10];
		delay100ms();
		P3_0 = !P3_0;
		P3_1 = !P3_1;
		P2 = lable[cnt % 10];
		delay100ms();
	}
}