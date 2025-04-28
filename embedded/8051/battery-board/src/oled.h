#ifndef __OLED_H__
#define __OLED_H__
 
void OLED_Clear(void);
void OLED_Init(void);
 
void OLED_ShowChar(unsigned char x,unsigned char y,unsigned char chr,unsigned char Char_Size);
void OLED_ShowNum(unsigned char x,unsigned char y,unsigned int num,unsigned char len,unsigned char  num_size);
void OLED_ShowString(unsigned char x,unsigned char y,unsigned char *chr,unsigned char Char_Size);
void OLED_ShowCHinese(unsigned char x,unsigned char y,unsigned char Position);
void OLED_ShowFloat(unsigned char x, unsigned char y, float num, unsigned char len, unsigned char num_size);
 
#endif