#include <reg51.h>

void delay(void) {
	int i = 0;
	for(;i < 10000;i++);
}

sbit P0_4 = P0^4;

int main() {
	P2 = 0xFF;
	while(1) {
	}
}