#include <reg51.h>

void delay(void) {
	int i;
	for(i = 0;i < 10000;i++);
}

void updatePorts(short val) {
	P2 = val;
	P0 = val;
}

int main(void) {
	updatePorts(0xFE);
	while(1) {
		delay();
		P2 <<= 1;
		P2 |= 0x01;		
		P0 = P2;
		delay();
		if(P2 == 0x7F) {
			updatePorts(0xFE);
		}
	}
	return 0;
}