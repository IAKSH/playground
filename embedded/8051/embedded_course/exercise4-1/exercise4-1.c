#include <reg51.h>

unsigned char count = 0;
unsigned char table[10] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};

void delay(unsigned int t) {
	unsigned int j = 0;
	for(;t > 0;t--)
		for(j = 0;j < 125;j++);
}

int main() {
	while(1) {
		P0 = ~table[count % 10];
		P3 = ~table[count / 10];
		delay(500);
		count = ++count % 100;
	}
}