#ifndef __I2C_H__
#define __I2C_H__
 
void Wait();
void Start();
void Send_Data(unsigned char dat);
void RecvACK();
unsigned char Recv_Data();
void SendACK();
void SendNAK();
void Stop();
 
#endif