#include <reg51.h>

unsigned char cnt = 0;
unsigned char table[10] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};
sbit PIN_BUTTON = P1^7;

// 12MHz
void delay1ms() {
	unsigned char i, j;

	i = 2;
	j = 239;
	do {
		while (--j);
	} while (--i);
}

// 12MHz
void delay10ms() {
	unsigned char i, j;

	i = 20;
	j = 113;
	do {
		while (--j);
	} while (--i);
}

int main() {
	while(1) {
		P3 = 0x01;
		P2 = table[cnt / 10];
		delay1ms();
		P3 = 0x02;
		P2 = table[cnt % 10];
		delay1ms();
		if(!PIN_BUTTON) {
			delay10ms();
			if(PIN_BUTTON)
				cnt = ++cnt % 100;
		}
	}
}