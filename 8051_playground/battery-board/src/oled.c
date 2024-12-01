#include "i2c.h"
#include "oledfont.h"
 
void Oled_Write_Cmd(char Command) {
	Start();
	Send_Data(0x78);
	RecvACK();
	Send_Data(0x00);
	RecvACK();
	Send_Data(Command);
	RecvACK();
	Stop();
}

void Oled_Write_Data(char Data) {
	Start();
	Send_Data(0x78);
	RecvACK();
	Send_Data(0x40);
	RecvACK();
	Send_Data(Data);
	RecvACK();
	Stop();
}
 
void OLED_Set_Pos(unsigned char x, unsigned char y) {
	Oled_Write_Cmd(0xb0+y);
	Oled_Write_Cmd(((x&0xf0)>>4)|0x10);
	Oled_Write_Cmd((x&0x0f)); 
}   	  

void OLED_Clear(void) {  
	unsigned char i,n;		    
	for(i=0;i<4;i++)  
	{  
		Oled_Write_Cmd(0xb0+i);
		Oled_Write_Cmd(0x00);
		Oled_Write_Cmd(0x10);  
		for(n=0;n<128;n++)Oled_Write_Data(0);
	} 
}
 
void OLED_ShowChar(unsigned char x,unsigned char y,unsigned char chr,unsigned char Char_Size) {      	
	unsigned char c=0,i=0;	
		c=chr-' ';		
		if(x>128-1){x=0;y=y+2;}
		if(Char_Size ==16)
			{
			OLED_Set_Pos(x,y);	
			for(i=0;i<8;i++)
			Oled_Write_Data(F8X16[c*16+i]);
			OLED_Set_Pos(x,y+1);
			for(i=0;i<8;i++)
			Oled_Write_Data(F8X16[c*16+i+8]);
			}
			else {	
				OLED_Set_Pos(x,y);
				for(i=0;i<6;i++)
				Oled_Write_Data(F6x8[c][i]);	
			}
}

void OLED_ShowString(unsigned char x,unsigned char y,unsigned char *chr,unsigned char Char_Size) {
	unsigned char j=0;
	while (chr[j]!='\0')
	{		OLED_ShowChar(x,y,chr[j],Char_Size);
			x+=8;
		if(x>120){x=0;y+=2;}
			j++;
	}
}
 
unsigned int oled_pow(unsigned char m,unsigned char n) {
	unsigned int result=1;	 
	while(n--)result*=m;    
	return result;
}			

void OLED_ShowNum(unsigned char x,unsigned char y,unsigned int num,unsigned char len,unsigned char  num_size) {         	
	unsigned char t,temp;
	unsigned char enshow=0;						   
	for(t=0;t<len;t++)
	{
		temp=(num/oled_pow(10,len-t-1))%10;
		if(enshow==0&&t<(len-1))
		{
			if(temp==0)
			{
				if(num_size == 8) OLED_ShowChar(x+(num_size/2+2)*t,y,' ',num_size);
				else if(num_size == 16) OLED_ShowChar(x+(num_size/2)*t,y,' ',num_size);
				continue;
			}else enshow=1; 
		 	 
		}
			if(num_size == 8) 	OLED_ShowChar(x+(num_size/2+2)*t,y,temp+'0',num_size);
			else if(num_size == 16) 	OLED_ShowChar(x+(num_size/2)*t,y,temp+'0',num_size);
	}
} 

void OLED_ShowCHinese(unsigned char x,unsigned char y,unsigned char Position) {      			    
	unsigned char t,adder=0;
	OLED_Set_Pos(x,y);	
    for(t=0;t<16;t++)
		{
				Oled_Write_Data(Hzk[2*Position][t]);
				adder+=1;
     }	
		OLED_Set_Pos(x,y+1);	
    for(t=0;t<16;t++)
			{	
				Oled_Write_Data(Hzk[2*Position+1][t]);
				adder+=1;
      }					
}
 
void OLED_Init(void) {	
	Oled_Write_Cmd(0xAE);
	Oled_Write_Cmd(0xD5);
	Oled_Write_Cmd(0X80);
	Oled_Write_Cmd(0xA8);
	Oled_Write_Cmd(0x1F);
	Oled_Write_Cmd(0xD3);
	Oled_Write_Cmd(0x00);
		
	Oled_Write_Cmd(0x40);
	
	Oled_Write_Cmd(0x8D);
	Oled_Write_Cmd(0x14);
	
	Oled_Write_Cmd(0x20);
	Oled_Write_Cmd(0x02);
	Oled_Write_Cmd(0xA1);
	Oled_Write_Cmd(0xC8);
	
	Oled_Write_Cmd(0xDA);
	Oled_Write_Cmd(0x02);
	
	Oled_Write_Cmd(0x81);
	Oled_Write_Cmd(0x8f);
	
	Oled_Write_Cmd(0xD9);
	Oled_Write_Cmd(0xf1);
	Oled_Write_Cmd(0xDB);
	Oled_Write_Cmd(0x40);
 
	Oled_Write_Cmd(0xA4);
	Oled_Write_Cmd(0xA6);
	
	Oled_Write_Cmd(0x2E);
 
	Oled_Write_Cmd(0xAF);
}  

int round(float num) {
	return (num >= 0) ? (int)(num + 0.5) : (int)(num - 0.5);
}

void OLED_ShowFloat(unsigned char x, unsigned char y, float num, unsigned char len, unsigned char num_size) {
    unsigned char i;
	unsigned char j;
    unsigned int int_part = (unsigned int)num;
    float fractional_part = num - int_part;
    unsigned int frac_as_int;

    if (int_part == 0) {
        OLED_ShowChar(x, y, '0', num_size);
        x += 8;
    } else {
        unsigned char buffer[10];
        i = 0;
        while (int_part > 0) {
            buffer[i++] = int_part % 10 + '0';
            int_part /= 10;
        }
        for (j = 0; j < i; j++) {
            OLED_ShowChar(x, y, buffer[i - j - 1], num_size);
            x += 8;
        }
    }

    OLED_ShowChar(x, y, '.', num_size);
    x += 8;

    for (i = 0; i < len; i++) {
        fractional_part *= 10;
    }
    frac_as_int = (unsigned int)round(fractional_part);
    if (frac_as_int == 0) {
        for (i = 0; i < len; i++) {
            OLED_ShowChar(x, y, '0', num_size);
            x += 8;
        }
    } else {
        unsigned char buffer[10];
        i = 0;
        while (frac_as_int > 0) {
            buffer[i++] = frac_as_int % 10 + '0';
            frac_as_int /= 10;
        }
        while (i < len) {
            buffer[i++] = '0';
        }
        for (j = 0; j < i; j++) {
            OLED_ShowChar(x, y, buffer[i - j - 1], num_size);
            x += 8;
        }
    }
}
