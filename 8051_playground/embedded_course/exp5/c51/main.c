#include <reg51.h>

char timer_cycle = 0;
char seconds = 0;
char table[]={0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};

void resetTimer0() {
	TH0 = 0x3c;
	TL0 = 0xb0;
}

void initTimer0() {
	TMOD = 0x01;
	resetTimer0();
	TR0 = 1;
}

void initTimer0Inter() {
	ET0 = 1;
	EA = 1;
}

void onTimer0Inter() interrupt 1 {
	resetTimer0();
	if(++timer_cycle == 20) {
		timer_cycle = 0;
		++seconds;
	}
	if(seconds == 60)
		seconds = 0;
	
	P0 = table[seconds / 10];
	P2 = table[seconds % 10];
}

int main() {
	P0 = P2 = table[0];
	
	initTimer0();
	initTimer0Inter();
	
	while(1);
}