#include <reg51.h>

unsigned char table[] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};
unsigned char count = 0;

void initUART() {
	PCON = 0x80;
	SCON = 0x90;
}

void sendChar(char c) {
	SBUF = c;
	while(!TI);
	TI = 0;
}

void waitAct() {
	while(!RI);
	RI = 0;
}

void delay(unsigned int t) {
	unsigned int j = 0;
	for(;t > 0;t--)
		for(j = 0;j < 125;j++);
}

int main() {
	initUART();
	while(1) {
		ACC = count;
		TB8 = P;
		sendChar(count);
		waitAct();
		if(!RB8) {
			P2 = table[count];
			count = ++count % 10;
			delay(500);
		}
	}
}