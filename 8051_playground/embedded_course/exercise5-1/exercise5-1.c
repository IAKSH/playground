#include <reg51.h>

unsigned char cnt = 0;
unsigned char table[10] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};

void exInter0Serv() interrupt 0 {
	cnt = ++cnt % 100;
	P0 = table[cnt / 10];
	P2 = table[cnt % 10];
}
	
void initInterrupt() {
	EX0 = 1;
	IT0 = 1;
	EA = 1;
}

void initPorts() {
	P0 = table[cnt / 10];
	P2 = table[cnt % 10];
}
	
int main() {
	initPorts();
	initInterrupt();
	while(1);
}