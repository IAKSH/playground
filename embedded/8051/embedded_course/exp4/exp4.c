#include <reg51.h>

sbit PIN_D1 = P0^4;
unsigned char i = 0;
unsigned char table[16] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f,0x77,0x7c,0x39,0x5e,0x79,0x71};

void initPorts() {
	PIN_D1 = 1;
	P2 = 0x00;
}

void initInterrupt() {
	EX0 = 1;
	EX1 = 1;
	IT0 = 1;
	IT1 = 1;
	EA = 1;
}

void exInter0Handler() interrupt 0 {
	PIN_D1 = !PIN_D1;
}

void exInter1Handler1() interrupt 2 {
	i = ++i % 16;
	P2 = table[i];
}

int main() {
	initPorts();
	initInterrupt();
	while(1);
}