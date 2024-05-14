#include <reg51.h>

unsigned char receive;
unsigned char table[] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};

void initUART() {
	PCON = 0x80;
	SCON = 0x90;
	ES = 1;
	EA = 1;
}

void uartIsr() interrupt 4 {
	receive = SBUF;
	RI = 0;
	
	//for test
	//++receive;
	
	ACC = receive;
	TB8 = (P != RB8);
	
	SBUF = receive;
	while(!TI);
	TI = 0;
	P2 = table[receive];
}

int main() {
	initUART();
	while(1);
}