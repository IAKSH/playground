#include "stc8g.h"

void Wait() {
    while(!(I2CMSST & 0x40));
    I2CMSST &= ~0x40;
}

void Start() {
    I2CMSCR = 0x01; // ??START??
    Wait();
}

void Send_Data(unsigned char dat) {
    I2CTXD = dat; // ?????????
    I2CMSCR = 0x02; // ??SEND??
    Wait();
}

void RecvACK() {
    I2CMSCR = 0x03; // ???ACK??
    Wait();
}

unsigned char Recv_Data() {
    I2CMSCR = 0x04; // ??RECV??
    Wait();
    return I2CRXD;
}

void SendACK() {
    I2CMSST = 0x00; // ??ACK??
    I2CMSCR = 0x05; // ??ACK??
    Wait();
}

void SendNAK() {
    I2CMSST = 0x01; // ??NAK??
    I2CMSCR = 0x05; // ??ACK??
    Wait();
}

void Stop() {
    I2CMSCR = 0x06; // ??STOP??
    Wait();
}

void I2C_Init() {
    P5M1 |= 0x30;
    P5M0 &= ~0x30;
    
    // 指定使用I2C2
    // 也许可以删掉
	P_SW2 &= ~(1 << 5);
	P_SW2 |= (1 << 4);
}
