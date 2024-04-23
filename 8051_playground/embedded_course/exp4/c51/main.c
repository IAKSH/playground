#include <reg51.h>

sbit D1 = P0^4;
char i = 0;
char led_mod[] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,
	0x07,0x7f,0x6f,0x77,0x7c,0x39,0x5e,0x79,0x71,0x00};

void delay(unsigned int t) {
	unsigned char j = 250;
	for(;t > 0;t--)
		for(;j > 0;j--);
}

void interKey1() interrupt 0 {
	D1 = !D1;
}

void interKey2() interrupt 2 {
	++i;
	i %= 16;
}

int main() {
	P0 = 0xFF;
	P3 = 0xFF;
	
	TCON = 0x05;
	IE = 0x85;
	
	while(1) {
		P2 = led_mod[i];
	}
}

