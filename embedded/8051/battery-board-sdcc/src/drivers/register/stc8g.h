#ifndef __STC8G_H__
#define __STC8G_H__

__sfr __at(0x80) P0;
__sbit __at(0x80) P00;
__sbit __at(0x81) P01;
__sbit __at(0x82) P02;
__sbit __at(0x83) P03;
__sbit __at(0x84) P04;
__sbit __at(0x85) P05;
__sbit __at(0x86) P06;
__sbit __at(0x87) P07;

__sfr __at(0x81) SP;
__sfr __at(0x82) DPL;
__sfr __at(0x83) DPH;
__sfr __at(0x84) S4CON;
__sfr __at(0x85) S4BUF;
__sfr __at(0x87) PCON;

__sfr __at(0x88) TCON;
__sbit __at(0x8f) TF1;
__sbit __at(0x8e) TR1;
__sbit __at(0x8d) TF0;
__sbit __at(0x8c) TR0;
__sbit __at(0x8b) IE1;
__sbit __at(0x8a) IT1;
__sbit __at(0x89) IE0;
__sbit __at(0x88) IT0;

__sfr __at(0x89) TMOD;
__sfr __at(0x8A) TL0;
__sfr __at(0x8B) TL1;
__sfr __at(0x8C) TH0;
__sfr __at(0x8D) TH1;
__sfr __at(0x8E) AUXR;
__sfr __at(0x8F) INTCLKO;

__sfr __at(0x90) P1;
__sbit __at(0x90) P10;
__sbit __at(0x91) P11;
__sbit __at(0x92) P12;
__sbit __at(0x93) P13;
__sbit __at(0x94) P14;
__sbit __at(0x95) P15;
__sbit __at(0x96) P16;
__sbit __at(0x97) P17;

__sfr __at(0x91) P1M1;
__sfr __at(0x92) P1M0;
__sfr __at(0x93) P0M1;
__sfr __at(0x94) P0M0;
__sfr __at(0x95) P2M1;
__sfr __at(0x96) P2M0;

__sfr __at(0x98) SCON;
__sbit __at(0x9f) SM0;
__sbit __at(0x9e) SM1;
__sbit __at(0x9d) SM2;
__sbit __at(0x9c) REN;
__sbit __at(0x9b) TB8;
__sbit __at(0x9a) RB8;
__sbit __at(0x99) TI;
__sbit __at(0x98) RI;

__sfr __at(0x99) SBUF;
__sfr __at(0x9A) S2CON;
__sfr __at(0x9B) S2BUF;
__sfr __at(0x9D) IRCBAND;
__sfr __at(0x9E) LIRTRIM;
__sfr __at(0x9F) IRTRIM;

__sfr __at(0xA0) P2;
__sbit __at(0xa0) P20;
__sbit __at(0xa1) P21;
__sbit __at(0xa2) P22;
__sbit __at(0xa3) P23;
__sbit __at(0xa4) P24;
__sbit __at(0xa5) P25;
__sbit __at(0xa6) P26;
__sbit __at(0xa7) P27;

__sfr __at(0xA2) P_SW1;

__sfr __at(0xA8) IE;
__sbit __at(0xaf) EA;
__sbit __at(0xae) ELVD;
__sbit __at(0xad) EADC;
__sbit __at(0xac) ES;
__sbit __at(0xab) ET1;
__sbit __at(0xaa) EX1;
__sbit __at(0xa9) ET0;
__sbit __at(0xa8) EX0;

__sfr __at(0xA9) SADDR;
__sfr __at(0xAA) WKTCL;
__sfr __at(0xAB) WKTCH;
__sfr __at(0xAC) S3CON;
__sfr __at(0xAD) S3BUF;
__sfr __at(0xAE) TA;
__sfr __at(0xAF) IE2;

__sfr __at(0xB0) P3;
__sbit __at(0xb0) P30;
__sbit __at(0xb1) P31;
__sbit __at(0xb2) P32;
__sbit __at(0xb3) P33;
__sbit __at(0xb4) P34;
__sbit __at(0xb5) P35;
__sbit __at(0xb6) P36;
__sbit __at(0xb7) P37;

__sbit __at(0xb7) RD;
__sbit __at(0xb6) WR;
__sbit __at(0xb5) T1;
__sbit __at(0xb4) T0;
__sbit __at(0xb3) INT1;
__sbit __at(0xb2) INT0;
__sbit __at(0xb1) TXD;
__sbit __at(0xb0) RXD;

__sfr __at(0xB1) P3M1;
__sfr __at(0xB2) P3M0;
__sfr __at(0xB3) P4M1;
__sfr __at(0xB4) P4M0;
__sfr __at(0xB5) IP2;
__sfr __at(0xB6) IP2H;
__sfr __at(0xB7) IPH;

__sfr __at(0xB8) IP;
__sbit __at(0xbf) PPCA;
__sbit __at(0xbe) PLVD;
__sbit __at(0xbd) PADC;
__sbit __at(0xbc) PS;
__sbit __at(0xbb) PT1;
__sbit __at(0xba) PX1;
__sbit __at(0xb9) PT0;
__sbit __at(0xb8) PX0;

__sfr __at(0xB9) SADEN;
__sfr __at(0xBA) P_SW2;
__sfr __at(0xBC) ADC_CONTR;
__sfr __at(0xBD) ADC_RES;
__sfr __at(0xBE) ADC_RESL;

__sfr __at(0xC0) P4;
__sbit __at(0xc0) P40;
__sbit __at(0xc1) P41;
__sbit __at(0xc2) P42;
__sbit __at(0xc3) P43;
__sbit __at(0xc4) P44;
__sbit __at(0xc5) P45;
__sbit __at(0xc6) P46;
__sbit __at(0xc7) P47;

__sfr __at(0xC1) WDT_CONTR;
__sfr __at(0xC2) IAP_DATA;
__sfr __at(0xC3) IAP_ADDRH;
__sfr __at(0xC4) IAP_ADDRL;
__sfr __at(0xC5) IAP_CMD;
__sfr __at(0xC6) IAP_TRIG;
__sfr __at(0xC7) IAP_CONTR;

__sfr __at(0xC8) P5;
__sbit __at(0xc8) P50;
__sbit __at(0xc9) P51;
__sbit __at(0xca) P52;
__sbit __at(0xcb) P53;
__sbit __at(0xcc) P54;
__sbit __at(0xcd) P55;
__sbit __at(0xce) P56;
__sbit __at(0xcf) P57;

__sfr __at(0xC9) P5M1;
__sfr __at(0xCA) P5M0;
__sfr __at(0xcb) P6M1;
__sfr __at(0xcc) P6M0;
__sfr __at(0xCD) SPSTAT;
__sfr __at(0xCE) SPCTL;
__sfr __at(0xCF) SPDAT;

__sfr __at(0xD0) PSW;
__sbit __at(0xd7) CY;
__sbit __at(0xd6) AC;
__sbit __at(0xd5) F0;
__sbit __at(0xd4) RS1;
__sbit __at(0xd3) RS0;
__sbit __at(0xd2) OV;
__sbit __at(0xd1) F1;
__sbit __at(0xd0) P;

__sfr __at(0xD1) T4T3M;
__sfr __at(0xD2) T4H;
__sfr __at(0xD3) T4L;
__sfr __at(0xD4) T3H;
__sfr __at(0xD5) T3L;
__sfr __at(0xD6) T2H;
__sfr __at(0xD7) T2L;

__sfr __at(0xD1) T3T4M;
__sfr __at(0xD2) TH4;
__sfr __at(0xD3) TL4;
__sfr __at(0xD4) TH3;
__sfr __at(0xD5) TL3;
__sfr __at(0xD6) TH2;
__sfr __at(0xD7) TL2;

__sfr __at(0xD8) CCON;
__sbit __at(0xdf) CF;
__sbit __at(0xde) CR;
__sbit __at(0xda) CCF2;
__sbit __at(0xd9) CCF1;
__sbit __at(0xd8) CCF0;

__sfr __at(0xD9) CMOD;
__sfr __at(0xDA) CCAPM0;
__sfr __at(0xDB) CCAPM1;
__sfr __at(0xDC) CCAPM2;
__sfr __at(0xDE) ADCCFG;
__sfr __at(0xDF) IP3;
__sfr __at(0xE0) ACC;
__sfr __at(0xe1) P7M1;
__sfr __at(0xe2) P7M0;
__sfr __at(0xE3) DPS;
__sfr __at(0xE4) DPL1;
__sfr __at(0xE5) DPH1;
__sfr __at(0xE6) CMPCR1;
__sfr __at(0xE7) CMPCR2;

__sfr __at(0xe8) P6;
__sbit __at(0xe8) P60;
__sbit __at(0xe9) P61;
__sbit __at(0xea) P62;
__sbit __at(0xeb) P63;
__sbit __at(0xec) P64;
__sbit __at(0xed) P65;
__sbit __at(0xee) P66;
__sbit __at(0xef) P67;

__sfr __at(0xE9) CL;
__sfr __at(0xEA) CCAP0L;
__sfr __at(0xEB) CCAP1L;
__sfr __at(0xEC) CCAP2L;
__sfr __at(0xEE) IP3H;
__sfr __at(0xEF) AUXINTIF;
__sfr __at(0xF0) B;
__sfr __at(0xF1) PWMSET;
__sfr __at(0xF2) PCA_PWM0;
__sfr __at(0xF3) PCA_PWM1;
__sfr __at(0xF4) PCA_PWM2;
__sfr __at(0xF5) IAP_TPS;
__sfr __at(0xF6) PWMCFG01;
__sfr __at(0xF7) PWMCFG23;

__sfr __at(0xf8) P7;
__sbit __at(0xf8) P70;
__sbit __at(0xf9) P71;
__sbit __at(0xfa) P72;
__sbit __at(0xfb) P73;
__sbit __at(0xfc) P74;
__sbit __at(0xfd) P75;
__sbit __at(0xfe) P76;
__sbit __at(0xff) P77;

__sfr __at(0xF9) CH;
__sfr __at(0xFA) CCAP0H;
__sfr __at(0xFB) CCAP1H;
__sfr __at(0xFC) CCAP2H;
__sfr __at(0xFE) PWMCFG45;
__sfr __at(0xFF) RSTCFG;

//如下特殊功能寄存器位于扩展RAM区域
//访问这些寄存器,需先将P_SW2的BIT7设置为1,才可正常读写

/////////////////////////////////////////////////
//FF00H-FFFFH
/////////////////////////////////////////////////

#define PWM0C (*(volatile unsigned int __xdata *)0xff00)
#define PWM0CH (*(volatile unsigned char __xdata *)0xff00)
#define PWM0CL (*(volatile unsigned char __xdata *)0xff01)
#define PWM0CKS (*(volatile unsigned char __xdata *)0xff02)
#define PWM0TADC (*(volatile unsigned int __xdata *)0xff03)
#define PWM0TADCH (*(volatile unsigned char __xdata *)0xff03)
#define PWM0TADCL (*(volatile unsigned char __xdata *)0xff04)
#define PWM0IF (*(volatile unsigned char __xdata *)0xff05)
#define PWM0FDCR (*(volatile unsigned char __xdata *)0xff06)
#define PWM00T1 (*(volatile unsigned int __xdata *)0xff10)
#define PWM00T1L (*(volatile unsigned char __xdata *)0xff11)
#define PWM00T2 (*(volatile unsigned int __xdata *)0xff12)
#define PWM00T2H (*(volatile unsigned char __xdata *)0xff12)
#define PWM00T2L (*(volatile unsigned char __xdata *)0xff13)
#define PWM00CR (*(volatile unsigned char __xdata *)0xff14)
#define PWM00HLD (*(volatile unsigned char __xdata *)0xff15)
#define PWM01T1 (*(volatile unsigned int __xdata *)0xff18)
#define PWM01T1H (*(volatile unsigned char __xdata *)0xff18)
#define PWM01T1L (*(volatile unsigned char __xdata *)0xff19)
#define PWM01T2 (*(volatile unsigned int __xdata *)0xff1a)
#define PWM01T2H (*(volatile unsigned char __xdata *)0xff1a)
#define PWM01T2L (*(volatile unsigned char __xdata *)0xff1b)
#define PWM01CR (*(volatile unsigned char __xdata *)0xff1c)
#define PWM01HLD (*(volatile unsigned char __xdata *)0xff1d)
#define PWM02T1 (*(volatile unsigned int __xdata *)0xff20)
#define PWM02T1H (*(volatile unsigned char __xdata *)0xff20)
#define PWM02T1L (*(volatile unsigned char __xdata *)0xff21)
#define PWM02T2 (*(volatile unsigned int __xdata *)0xff22)
#define PWM02T2H (*(volatile unsigned char __xdata *)0xff22)
#define PWM02T2L (*(volatile unsigned char __xdata *)0xff23)
#define PWM02CR (*(volatile unsigned char __xdata *)0xff24)
#define PWM02HLD (*(volatile unsigned char __xdata *)0xff25)
#define PWM03T1 (*(volatile unsigned int __xdata *)0xff28)
#define PWM03T1H (*(volatile unsigned char __xdata *)0xff28)
#define PWM03T1L (*(volatile unsigned char __xdata *)0xff29)
#define PWM03T2 (*(volatile unsigned int __xdata *)0xff2a)
#define PWM03T2H (*(volatile unsigned char __xdata *)0xff2a)
#define PWM03T2L (*(volatile unsigned char __xdata *)0xff2b)
#define PWM03CR (*(volatile unsigned char __xdata *)0xff2c)
#define PWM03HLD (*(volatile unsigned char __xdata *)0xff2d)
#define PWM04T1 (*(volatile unsigned int __xdata *)0xff30)
#define PWM04T1H (*(volatile unsigned char __xdata *)0xff30)
#define PWM04T1L (*(volatile unsigned char __xdata *)0xff31)
#define PWM04T2 (*(volatile unsigned int __xdata *)0xff32)
#define PWM04T2H (*(volatile unsigned char __xdata *)0xff32)
#define PWM04T2L (*(volatile unsigned char __xdata *)0xff33)
#define PWM04CR (*(volatile unsigned char __xdata *)0xff34)
#define PWM04HLD (*(volatile unsigned char __xdata *)0xff35)
#define PWM05T1 (*(volatile unsigned int __xdata *)0xff38)
#define PWM05T1H (*(volatile unsigned char __xdata *)0xff38)
#define PWM05T1L (*(volatile unsigned char __xdata *)0xff39)
#define PWM05T2 (*(volatile unsigned int __xdata *)0xff3a)
#define PWM05T2H (*(volatile unsigned char __xdata *)0xff3a)
#define PWM05T2L (*(volatile unsigned char __xdata *)0xff3b)
#define PWM05CR (*(volatile unsigned char __xdata *)0xff3c)
#define PWM05HLD (*(volatile unsigned char __xdata *)0xff3d)
#define PWM06T1 (*(volatile unsigned int __xdata *)0xff40)
#define PWM06T1H (*(volatile unsigned char __xdata *)0xff40)
#define PWM06T1L (*(volatile unsigned char __xdata *)0xff41)
#define PWM06T2 (*(volatile unsigned int __xdata *)0xff42)
#define PWM06T2H (*(volatile unsigned char __xdata *)0xff42)
#define PWM06T2L (*(volatile unsigned char __xdata *)0xff43)
#define PWM06CR (*(volatile unsigned char __xdata *)0xff44)
#define PWM06HLD (*(volatile unsigned char __xdata *)0xff45)
#define PWM07T1 (*(volatile unsigned int __xdata *)0xff48)
#define PWM07T1H (*(volatile unsigned char __xdata *)0xff48)
#define PWM07T1L (*(volatile unsigned char __xdata *)0xff49)
#define PWM07T2 (*(volatile unsigned int __xdata *)0xff4a)
#define PWM07T2H (*(volatile unsigned char __xdata *)0xff4a)
#define PWM07T2L (*(volatile unsigned char __xdata *)0xff4b)
#define PWM07CR (*(volatile unsigned char __xdata *)0xff4c)
#define PWM07HLD (*(volatile unsigned char __xdata *)0xff4d)
#define PWM1C (*(volatile unsigned int __xdata *)0xff50)
#define PWM1CH (*(volatile unsigned char __xdata *)0xff50)
#define PWM1CL (*(volatile unsigned char __xdata *)0xff51)
#define PWM1CKS (*(volatile unsigned char __xdata *)0xff52)
#define PWM1IF (*(volatile unsigned char __xdata *)0xff55)
#define PWM1FDCR (*(volatile unsigned char __xdata *)0xff56)
#define PWM10T1 (*(volatile unsigned int __xdata *)0xff60)
#define PWM10T1H (*(volatile unsigned char __xdata *)0xff60)
#define PWM10T1L (*(volatile unsigned char __xdata *)0xff61)
#define PWM10T2 (*(volatile unsigned int __xdata *)0xff62)
#define PWM10T2H (*(volatile unsigned char __xdata *)0xff62)
#define PWM10T2L (*(volatile unsigned char __xdata *)0xff63)
#define PWM10CR (*(volatile unsigned char __xdata *)0xff64)
#define PWM10HLD (*(volatile unsigned char __xdata *)0xff65)
#define PWM11T1 (*(volatile unsigned int __xdata *)0xff68)
#define PWM11T1H (*(volatile unsigned char __xdata *)0xff68)
#define PWM11T1L (*(volatile unsigned char __xdata *)0xff69)
#define PWM11T2 (*(volatile unsigned int __xdata *)0xff6a)
#define PWM11T2H (*(volatile unsigned char __xdata *)0xff6a)
#define PWM11T2L (*(volatile unsigned char __xdata *)0xff6b)
#define PWM11CR (*(volatile unsigned char __xdata *)0xff6c)
#define PWM11HLD (*(volatile unsigned char __xdata *)0xff6d)
#define PWM12T1 (*(volatile unsigned int __xdata *)0xff70)
#define PWM12T1H (*(volatile unsigned char __xdata *)0xff70)
#define PWM12T1L (*(volatile unsigned char __xdata *)0xff71)
#define PWM12T2 (*(volatile unsigned int __xdata *)0xff72)
#define PWM12T2H (*(volatile unsigned char __xdata *)0xff72)
#define PWM12T2L (*(volatile unsigned char __xdata *)0xff73)
#define PWM12CR (*(volatile unsigned char __xdata *)0xff74)
#define PWM12HLD (*(volatile unsigned char __xdata *)0xff75)
#define PWM13T1 (*(volatile unsigned int __xdata *)0xff78)
#define PWM13T1H (*(volatile unsigned char __xdata *)0xff78)
#define PWM13T1L (*(volatile unsigned char __xdata *)0xff79)
#define PWM13T2 (*(volatile unsigned int __xdata *)0xff7a)
#define PWM13T2H (*(volatile unsigned char __xdata *)0xff7a)
#define PWM13T2L (*(volatile unsigned char __xdata *)0xff7b)
#define PWM13CR (*(volatile unsigned char __xdata *)0xff7c)
#define PWM13HLD (*(volatile unsigned char __xdata *)0xff7d)
#define PWM14T1 (*(volatile unsigned int __xdata *)0xff80)
#define PWM14T1H (*(volatile unsigned char __xdata *)0xff80)
#define PWM14T1L (*(volatile unsigned char __xdata *)0xff81)
#define PWM14T2 (*(volatile unsigned int __xdata *)0xff82)
#define PWM14T2H (*(volatile unsigned char __xdata *)0xff82)
#define PWM14T2L (*(volatile unsigned char __xdata *)0xff83)
#define PWM14CR (*(volatile unsigned char __xdata *)0xff84)
#define PWM14HLD (*(volatile unsigned char __xdata *)0xff85)
#define PWM15T1 (*(volatile unsigned int __xdata *)0xff88)
#define PWM15T1H (*(volatile unsigned char __xdata *)0xff88)
#define PWM15T1L (*(volatile unsigned char __xdata *)0xff89)
#define PWM15T2 (*(volatile unsigned int __xdata *)0xff8a)
#define PWM15T2H (*(volatile unsigned char __xdata *)0xff8a)
#define PWM15T2L (*(volatile unsigned char __xdata *)0xff8b)
#define PWM15CR (*(volatile unsigned char __xdata *)0xff8c)
#define PWM15HLD (*(volatile unsigned char __xdata *)0xff8d)
#define PWM16T1 (*(volatile unsigned int __xdata *)0xff90)
#define PWM16T1H (*(volatile unsigned char __xdata *)0xff90)
#define PWM16T1L (*(volatile unsigned char __xdata *)0xff91)
#define PWM16T2 (*(volatile unsigned int __xdata *)0xff92)
#define PWM16T2H (*(volatile unsigned char __xdata *)0xff92)
#define PWM16T2L (*(volatile unsigned char __xdata *)0xff93)
#define PWM16CR (*(volatile unsigned char __xdata *)0xff94)
#define PWM16HLD (*(volatile unsigned char __xdata *)0xff95)
#define PWM17T1 (*(volatile unsigned int __xdata *)0xff98)
#define PWM17T1H (*(volatile unsigned char __xdata *)0xff98)
#define PWM17T1L (*(volatile unsigned char __xdata *)0xff99)
#define PWM17T2 (*(volatile unsigned int __xdata *)0xff9a)
#define PWM17T2H (*(volatile unsigned char __xdata *)0xff9a)
#define PWM17T2L (*(volatile unsigned char __xdata *)0xff9b)
#define PWM17CR (*(volatile unsigned char __xdata *)0xff9c)
#define PWM17HLD (*(volatile unsigned char __xdata *)0xff9d)
#define PWM2C (*(volatile unsigned int __xdata *)0xffa0)
#define PWM2CH (*(volatile unsigned char __xdata *)0xffa0)
#define PWM2CL (*(volatile unsigned char __xdata *)0xffa1)
#define PWM2CKS (*(volatile unsigned char __xdata *)0xffa2)
#define PWM2TADC (*(volatile unsigned int __xdata *)0xffa3)
#define PWM2TADCH (*(volatile unsigned char __xdata *)0xffa3)
#define PWM2TADCL (*(volatile unsigned char __xdata *)0xffa4)
#define PWM2IF (*(volatile unsigned char __xdata *)0xffa5)
#define PWM2FDCR (*(volatile unsigned char __xdata *)0xffa6)
#define PWM20T1 (*(volatile unsigned int __xdata *)0xffb0)
#define PWM20T1H (*(volatile unsigned char __xdata *)0xffb0)
#define PWM20T1L (*(volatile unsigned char __xdata *)0xffb1)
#define PWM20T2 (*(volatile unsigned int __xdata *)0xffb2)
#define PWM20T2H (*(volatile unsigned char __xdata *)0xffb2)
#define PWM20T2L (*(volatile unsigned char __xdata *)0xffb3)
#define PWM20CR (*(volatile unsigned char __xdata *)0xffb4)
#define PWM20HLD (*(volatile unsigned char __xdata *)0xffb5)
#define PWM21T1 (*(volatile unsigned int __xdata *)0xffb8)
#define PWM21T1H (*(volatile unsigned char __xdata *)0xffb8)
#define PWM21T1L (*(volatile unsigned char __xdata *)0xffb9)
#define PWM21T2 (*(volatile unsigned int __xdata *)0xffba)
#define PWM21T2H (*(volatile unsigned char __xdata *)0xffba)
#define PWM21T2L (*(volatile unsigned char __xdata *)0xffbb)
#define PWM21CR (*(volatile unsigned char __xdata *)0xffbc)
#define PWM21HLD (*(volatile unsigned char __xdata *)0xffbd)
#define PWM22T1 (*(volatile unsigned int __xdata *)0xffc0)
#define PWM22T1H (*(volatile unsigned char __xdata *)0xffc0)
#define PWM22T1L (*(volatile unsigned char __xdata *)0xffc1)
#define PWM22T2 (*(volatile unsigned int __xdata *)0xffc2)
#define PWM22T2H (*(volatile unsigned char __xdata *)0xffc2)
#define PWM22T2L (*(volatile unsigned char __xdata *)0xffc3)
#define PWM22CR (*(volatile unsigned char __xdata *)0xffc4)
#define PWM22HLD (*(volatile unsigned char __xdata *)0xffc5)
#define PWM23T1 (*(volatile unsigned int __xdata *)0xffc8)
#define PWM23T1H (*(volatile unsigned char __xdata *)0xffc8)
#define PWM23T1L (*(volatile unsigned char __xdata *)0xffc9)
#define PWM23T2 (*(volatile unsigned int __xdata *)0xffca)
#define PWM23T2H (*(volatile unsigned char __xdata *)0xffca)
#define PWM23T2L (*(volatile unsigned char __xdata *)0xffcb)
#define PWM23CR (*(volatile unsigned char __xdata *)0xffcc)
#define PWM23HLD (*(volatile unsigned char __xdata *)0xffcd)
#define PWM24T1 (*(volatile unsigned int __xdata *)0xffd0)
#define PWM24T1H (*(volatile unsigned char __xdata *)0xffd0)
#define PWM24T1L (*(volatile unsigned char __xdata *)0xffd1)
#define PWM24T2 (*(volatile unsigned int __xdata *)0xffd2)
#define PWM24T2H (*(volatile unsigned char __xdata *)0xffd2)
#define PWM24T2L (*(volatile unsigned char __xdata *)0xffd3)
#define PWM24CR (*(volatile unsigned char __xdata *)0xffd4)
#define PWM24HLD (*(volatile unsigned char __xdata *)0xffd5)
#define PWM25T1 (*(volatile unsigned int __xdata *)0xffd8)
#define PWM25T1H (*(volatile unsigned char __xdata *)0xffd8)
#define PWM25T1L (*(volatile unsigned char __xdata *)0xffd9)
#define PWM25T2 (*(volatile unsigned int __xdata *)0xffda)
#define PWM25T2H (*(volatile unsigned char __xdata *)0xffda)
#define PWM25T2L (*(volatile unsigned char __xdata *)0xffdb)
#define PWM25CR (*(volatile unsigned char __xdata *)0xffdc)
#define PWM25HLD (*(volatile unsigned char __xdata *)0xffdd)
#define PWM26T1 (*(volatile unsigned int __xdata *)0xffe0)
#define PWM26T1H (*(volatile unsigned char __xdata *)0xffe0)
#define PWM26T1L (*(volatile unsigned char __xdata *)0xffe1)
#define PWM26T2 (*(volatile unsigned int __xdata *)0xffe2)
#define PWM26T2H (*(volatile unsigned char __xdata *)0xffe2)
#define PWM26T2L (*(volatile unsigned char __xdata *)0xffe3)
#define PWM26CR (*(volatile unsigned char __xdata *)0xffe4)
#define PWM26HLD (*(volatile unsigned char __xdata *)0xffe5)
#define PWM27T1 (*(volatile unsigned int __xdata *)0xffe8)
#define PWM27T1H (*(volatile unsigned char __xdata *)0xffe8)
#define PWM27T1L (*(volatile unsigned char __xdata *)0xffe9)
#define PWM27T2 (*(volatile unsigned int __xdata *)0xffea)
#define PWM27T2H (*(volatile unsigned char __xdata *)0xffea)
#define PWM27T2L (*(volatile unsigned char __xdata *)0xffeb)
#define PWM27CR (*(volatile unsigned char __xdata *)0xffec)
#define PWM27HLD (*(volatile unsigned char __xdata *)0xffed)

/////////////////////////////////////////////////
//FE00H-FEFFH
/////////////////////////////////////////////////

#define CLKSEL (*(volatile unsigned char __xdata *)0xfe00)
#define CLKDIV (*(volatile unsigned char __xdata *)0xfe01)
#define HIRCCR (*(volatile unsigned char __xdata *)0xfe02)
#define XOSCCR (*(volatile unsigned char __xdata *)0xfe03)
#define IRC32KCR (*(volatile unsigned char __xdata *)0xfe04)
#define MCLKOCR (*(volatile unsigned char __xdata *)0xfe05)
#define IRCDB (*(volatile unsigned char __xdata *)0xfe06)
#define X32KCR (*(volatile unsigned char __xdata *)0xfe08)
                                    
#define P0PU (*(volatile unsigned char __xdata *)0xfe10)
#define P1PU (*(volatile unsigned char __xdata *)0xfe11)
#define P2PU (*(volatile unsigned char __xdata *)0xfe12)
#define P3PU (*(volatile unsigned char __xdata *)0xfe13)
#define P4PU (*(volatile unsigned char __xdata *)0xfe14)
#define P5PU (*(volatile unsigned char __xdata *)0xfe15)
#define P6PU (*(volatile unsigned char __xdata *)0xfe16)
#define P7PU (*(volatile unsigned char __xdata *)0xfe17)
#define P0NCS (*(volatile unsigned char __xdata *)0xfe18)
#define P1NCS (*(volatile unsigned char __xdata *)0xfe19)
#define P2NCS (*(volatile unsigned char __xdata *)0xfe1a)
#define P3NCS (*(volatile unsigned char __xdata *)0xfe1b)
#define P4NCS (*(volatile unsigned char __xdata *)0xfe1c)
#define P5NCS (*(volatile unsigned char __xdata *)0xfe1d)
#define P6NCS (*(volatile unsigned char __xdata *)0xfe1e)
#define P7NCS (*(volatile unsigned char __xdata *)0xfe1f)
#define P0SR (*(volatile unsigned char __xdata *)0xfe20)
#define P1SR (*(volatile unsigned char __xdata *)0xfe21)
#define P2SR (*(volatile unsigned char __xdata *)0xfe22)
#define P3SR (*(volatile unsigned char __xdata *)0xfe23)
#define P4SR (*(volatile unsigned char __xdata *)0xfe24)
#define P5SR (*(volatile unsigned char __xdata *)0xfe25)
#define P6SR (*(volatile unsigned char __xdata *)0xfe26)
#define P7SR (*(volatile unsigned char __xdata *)0xfe27)
#define P0DR (*(volatile unsigned char __xdata *)0xfe28)
#define P1DR (*(volatile unsigned char __xdata *)0xfe29)
#define P2DR (*(volatile unsigned char __xdata *)0xfe2a)
#define P3DR (*(volatile unsigned char __xdata *)0xfe2b)
#define P4DR (*(volatile unsigned char __xdata *)0xfe2c)
#define P5DR (*(volatile unsigned char __xdata *)0xfe2d)
#define P6DR (*(volatile unsigned char __xdata *)0xfe2e)
#define P7DR (*(volatile unsigned char __xdata *)0xfe2f)
#define P0IE (*(volatile unsigned char __xdata *)0xfe30)
#define P1IE (*(volatile unsigned char __xdata *)0xfe31)
#define P2IE (*(volatile unsigned char __xdata *)0xfe32)
#define P3IE (*(volatile unsigned char __xdata *)0xfe33)
#define P4IE (*(volatile unsigned char __xdata *)0xfe34)
#define P5IE (*(volatile unsigned char __xdata *)0xfe35)
#define P6IE (*(volatile unsigned char __xdata *)0xfe36)
#define P7IE (*(volatile unsigned char __xdata *)0xfe37)
                                    
#define RTCCR (*(volatile unsigned char __xdata *)0xfe60)
#define RTCCFG (*(volatile unsigned char __xdata *)0xfe61)
#define RTCIEN (*(volatile unsigned char __xdata *)0xfe62)
#define RTCIF (*(volatile unsigned char __xdata *)0xfe63)
#define ALAHOUR (*(volatile unsigned char __xdata *)0xfe64)
#define ALAMIN (*(volatile unsigned char __xdata *)0xfe65)
#define ALASEC (*(volatile unsigned char __xdata *)0xfe66)
#define ALASSEC (*(volatile unsigned char __xdata *)0xfe67)
#define INIYEAR (*(volatile unsigned char __xdata *)0xfe68)
#define INIMONTH (*(volatile unsigned char __xdata *)0xfe69)
#define INIDAY (*(volatile unsigned char __xdata *)0xfe6a)
#define INIHOUR (*(volatile unsigned char __xdata *)0xfe6b)
#define INIMIN (*(volatile unsigned char __xdata *)0xfe6c)
#define INISEC (*(volatile unsigned char __xdata *)0xfe6d)
#define INISSEC (*(volatile unsigned char __xdata *)0xfe6e)
#define YEAR (*(volatile unsigned char __xdata *)0xfe70)
#define MONTH (*(volatile unsigned char __xdata *)0xfe71)
#define DAY (*(volatile unsigned char __xdata *)0xfe72)
#define HOUR (*(volatile unsigned char __xdata *)0xfe73)
#define MIN (*(volatile unsigned char __xdata *)0xfe74)
#define SEC (*(volatile unsigned char __xdata *)0xfe75)
#define SSEC (*(volatile unsigned char __xdata *)0xfe76)
                                    
#define I2CCFG (*(volatile unsigned char __xdata *)0xfe80)
#define I2CMSCR (*(volatile unsigned char __xdata *)0xfe81)
#define I2CMSST (*(volatile unsigned char __xdata *)0xfe82)
#define I2CSLCR (*(volatile unsigned char __xdata *)0xfe83)
#define I2CSLST (*(volatile unsigned char __xdata *)0xfe84)
#define I2CSLADR (*(volatile unsigned char __xdata *)0xfe85)
#define I2CTXD (*(volatile unsigned char __xdata *)0xfe86)
#define I2CRXD (*(volatile unsigned char __xdata *)0xfe87)
#define I2CMSAUX (*(volatile unsigned char __xdata *)0xfe88)
                                    
#define TM2PS (*(volatile unsigned char __xdata *)0xfea2)
#define TM3PS (*(volatile unsigned char __xdata *)0xfea3)
#define TM4PS (*(volatile unsigned char __xdata *)0xfea4)
#define ADCTIM (*(volatile unsigned char __xdata *)0xfea8)
#define T3T4PS (*(volatile unsigned char __xdata *)0xfeac)

/////////////////////////////////////////////////
//FD00H-FDFFH
/////////////////////////////////////////////////

#define P0INTE (*(volatile unsigned char __xdata *)0xfd00)
#define P1INTE (*(volatile unsigned char __xdata *)0xfd01)
#define P2INTE (*(volatile unsigned char __xdata *)0xfd02)
#define P3INTE (*(volatile unsigned char __xdata *)0xfd03)
#define P4INTE (*(volatile unsigned char __xdata *)0xfd04)
#define P5INTE (*(volatile unsigned char __xdata *)0xfd05)
#define P6INTE (*(volatile unsigned char __xdata *)0xfd06)
#define P7INTE (*(volatile unsigned char __xdata *)0xfd07)
#define P0INTF (*(volatile unsigned char __xdata *)0xfd10)
#define P1INTF (*(volatile unsigned char __xdata *)0xfd11)
#define P2INTF (*(volatile unsigned char __xdata *)0xfd12)
#define P3INTF (*(volatile unsigned char __xdata *)0xfd13)
#define P4INTF (*(volatile unsigned char __xdata *)0xfd14)
#define P5INTF (*(volatile unsigned char __xdata *)0xfd15)
#define P6INTF (*(volatile unsigned char __xdata *)0xfd16)
#define P7INTF (*(volatile unsigned char __xdata *)0xfd17)
#define P0IM0 (*(volatile unsigned char __xdata *)0xfd20)
#define P1IM0 (*(volatile unsigned char __xdata *)0xfd21)
#define P2IM0 (*(volatile unsigned char __xdata *)0xfd22)
#define P3IM0 (*(volatile unsigned char __xdata *)0xfd23)
#define P4IM0 (*(volatile unsigned char __xdata *)0xfd24)
#define P5IM0 (*(volatile unsigned char __xdata *)0xfd25)
#define P6IM0 (*(volatile unsigned char __xdata *)0xfd26)
#define P7IM0 (*(volatile unsigned char __xdata *)0xfd27)
#define P0IM1 (*(volatile unsigned char __xdata *)0xfd30)
#define P1IM1 (*(volatile unsigned char __xdata *)0xfd31)
#define P2IM1 (*(volatile unsigned char __xdata *)0xfd32)
#define P3IM1 (*(volatile unsigned char __xdata *)0xfd33)
#define P4IM1 (*(volatile unsigned char __xdata *)0xfd34)
#define P5IM1 (*(volatile unsigned char __xdata *)0xfd35)
#define P6IM1 (*(volatile unsigned char __xdata *)0xfd36)
#define P7IM1 (*(volatile unsigned char __xdata *)0xfd37)
#define P0WKUE (*(volatile unsigned char __xdata *)0xfd40)
#define P1WKUE (*(volatile unsigned char __xdata *)0xfd41)
#define P2WKUE (*(volatile unsigned char __xdata *)0xfd42)
#define P3WKUE (*(volatile unsigned char __xdata *)0xfd43)
#define P4WKUE (*(volatile unsigned char __xdata *)0xfd44)
#define P5WKUE (*(volatile unsigned char __xdata *)0xfd45)
#define P6WKUE (*(volatile unsigned char __xdata *)0xfd46)
#define P7WKUE (*(volatile unsigned char __xdata *)0xfd47)
#define PINIPL (*(volatile unsigned char __xdata *)0xfd60)
#define PINIPH (*(volatile unsigned char __xdata *)0xfd61)

/////////////////////////////////////////////////
//FC00H-FCFFH
/////////////////////////////////////////////////

#define PWM3C (*(volatile unsigned int __xdata *)0xfc00)
#define PWM3CH (*(volatile unsigned char __xdata *)0xfc00)
#define PWM3CL (*(volatile unsigned char __xdata *)0xfc01)
#define PWM3CKS (*(volatile unsigned char __xdata *)0xfc02)
#define PWM3IF (*(volatile unsigned char __xdata *)0xfc05)
#define PWM3FDCR (*(volatile unsigned char __xdata *)0xfc06)
#define PWM30T1 (*(volatile unsigned int __xdata *)0xfc10)
#define PWM30T1H (*(volatile unsigned char __xdata *)0xfc10)
#define PWM30T1L (*(volatile unsigned char __xdata *)0xfc11)
#define PWM30T2 (*(volatile unsigned int __xdata *)0xfc12)
#define PWM30T2H (*(volatile unsigned char __xdata *)0xfc12)
#define PWM30T2L (*(volatile unsigned char __xdata *)0xfc13)
#define PWM30CR (*(volatile unsigned char __xdata *)0xfc14)
#define PWM30HLD (*(volatile unsigned char __xdata *)0xfc15)
#define PWM31T1 (*(volatile unsigned int __xdata *)0xfc18)
#define PWM31T1H (*(volatile unsigned char __xdata *)0xfc18)
#define PWM31T1L (*(volatile unsigned char __xdata *)0xfc19)
#define PWM31T2 (*(volatile unsigned int __xdata *)0xfc1a)
#define PWM31T2H (*(volatile unsigned char __xdata *)0xfc1a)
#define PWM31T2L (*(volatile unsigned char __xdata *)0xfc1b)
#define PWM31CR (*(volatile unsigned char __xdata *)0xfc1c)
#define PWM31HLD (*(volatile unsigned char __xdata *)0xfc1d)
#define PWM32T1 (*(volatile unsigned int __xdata *)0xfc20)
#define PWM32T1H (*(volatile unsigned char __xdata *)0xfc20)
#define PWM32T1L (*(volatile unsigned char __xdata *)0xfc21)
#define PWM32T2 (*(volatile unsigned int __xdata *)0xfc22)
#define PWM32T2H (*(volatile unsigned char __xdata *)0xfc22)
#define PWM32T2L (*(volatile unsigned char __xdata *)0xfc23)
#define PWM32CR (*(volatile unsigned char __xdata *)0xfc24)
#define PWM32HLD (*(volatile unsigned char __xdata *)0xfc25)
#define PWM33T1 (*(volatile unsigned int __xdata *)0xfc28)
#define PWM33T1H (*(volatile unsigned char __xdata *)0xfc28)
#define PWM33T1L (*(volatile unsigned char __xdata *)0xfc29)
#define PWM33T2 (*(volatile unsigned int __xdata *)0xfc2a)
#define PWM33T2H (*(volatile unsigned char __xdata *)0xfc2a)
#define PWM33T2L (*(volatile unsigned char __xdata *)0xfc2b)
#define PWM33CR (*(volatile unsigned char __xdata *)0xfc2c)
#define PWM33HLD (*(volatile unsigned char __xdata *)0xfc2d)
#define PWM34T1 (*(volatile unsigned int __xdata *)0xfc30)
#define PWM34T1H (*(volatile unsigned char __xdata *)0xfc30)
#define PWM34T1L (*(volatile unsigned char __xdata *)0xfc31)
#define PWM34T2 (*(volatile unsigned int __xdata *)0xfc32)
#define PWM34T2H (*(volatile unsigned char __xdata *)0xfc32)
#define PWM34T2L (*(volatile unsigned char __xdata *)0xfc33)
#define PWM34CR (*(volatile unsigned char __xdata *)0xfc34)
#define PWM34HLD (*(volatile unsigned char __xdata *)0xfc35)
#define PWM35T1 (*(volatile unsigned int __xdata *)0xfc38)
#define PWM35T1H (*(volatile unsigned char __xdata *)0xfc38)
#define PWM35T1L (*(volatile unsigned char __xdata *)0xfc39)
#define PWM35T2 (*(volatile unsigned int __xdata *)0xfc3a)
#define PWM35T2H (*(volatile unsigned char __xdata *)0xfc3a)
#define PWM35T2L (*(volatile unsigned char __xdata *)0xfc3b)
#define PWM35CR (*(volatile unsigned char __xdata *)0xfc3c)
#define PWM35HLD (*(volatile unsigned char __xdata *)0xfc3d)
#define PWM36T1 (*(volatile unsigned int __xdata *)0xfc40)
#define PWM36T1H (*(volatile unsigned char __xdata *)0xfc40)
#define PWM36T1L (*(volatile unsigned char __xdata *)0xfc41)
#define PWM36T2 (*(volatile unsigned int __xdata *)0xfc42)
#define PWM36T2H (*(volatile unsigned char __xdata *)0xfc42)
#define PWM36T2L (*(volatile unsigned char __xdata *)0xfc43)
#define PWM36CR (*(volatile unsigned char __xdata *)0xfc44)
#define PWM36HLD (*(volatile unsigned char __xdata *)0xfc45)
#define PWM37T1 (*(volatile unsigned int __xdata *)0xfc48)
#define PWM37T1H (*(volatile unsigned char __xdata *)0xfc48)
#define PWM37T1L (*(volatile unsigned char __xdata *)0xfc49)
#define PWM37T2 (*(volatile unsigned int __xdata *)0xfc4a)
#define PWM37T2H (*(volatile unsigned char __xdata *)0xfc4a)
#define PWM37T2L (*(volatile unsigned char __xdata *)0xfc4b)
#define PWM37CR (*(volatile unsigned char __xdata *)0xfc4c)
#define PWM37HLD (*(volatile unsigned char __xdata *)0xfc4d)
#define PWM4C (*(volatile unsigned int __xdata *)0xfc50)
#define PWM4CH (*(volatile unsigned char __xdata *)0xfc50)
#define PWM4CL (*(volatile unsigned char __xdata *)0xfc51)
#define PWM4CKS (*(volatile unsigned char __xdata *)0xfc52)
#define PWM4TADC (*(volatile unsigned int __xdata *)0xfc53)
#define PWM4TADCH (*(volatile unsigned char __xdata *)0xfc53)
#define PWM4TADCL (*(volatile unsigned char __xdata *)0xfc54)
#define PWM4IF (*(volatile unsigned char __xdata *)0xfc55)
#define PWM4FDCR (*(volatile unsigned char __xdata *)0xfc56)
#define PWM40T1 (*(volatile unsigned int __xdata *)0xfc60)
#define PWM40T1H (*(volatile unsigned char __xdata *)0xfc60)
#define PWM40T1L (*(volatile unsigned char __xdata *)0xfc61)
#define PWM40T2 (*(volatile unsigned int __xdata *)0xfc62)
#define PWM40T2H (*(volatile unsigned char __xdata *)0xfc62)
#define PWM40T2L (*(volatile unsigned char __xdata *)0xfc63)
#define PWM40CR (*(volatile unsigned char __xdata *)0xfc64)
#define PWM40HLD (*(volatile unsigned char __xdata *)0xfc65)
#define PWM41T1 (*(volatile unsigned int __xdata *)0xfc68)
#define PWM41T1H (*(volatile unsigned char __xdata *)0xfc68)
#define PWM41T1L (*(volatile unsigned char __xdata *)0xfc69)
#define PWM41T2 (*(volatile unsigned int __xdata *)0xfc6a)
#define PWM41T2H (*(volatile unsigned char __xdata *)0xfc6a)
#define PWM41T2L (*(volatile unsigned char __xdata *)0xfc6b)
#define PWM41CR (*(volatile unsigned char __xdata *)0xfc6c)
#define PWM41HLD (*(volatile unsigned char __xdata *)0xfc6d)
#define PWM42T1 (*(volatile unsigned int __xdata *)0xfc70)
#define PWM42T1H (*(volatile unsigned char __xdata *)0xfc70)
#define PWM42T1L (*(volatile unsigned char __xdata *)0xfc71)
#define PWM42T2 (*(volatile unsigned int __xdata *)0xfc72)
#define PWM42T2H (*(volatile unsigned char __xdata *)0xfc72)
#define PWM42T2L (*(volatile unsigned char __xdata *)0xfc73)
#define PWM42CR (*(volatile unsigned char __xdata *)0xfc74)
#define PWM42HLD (*(volatile unsigned char __xdata *)0xfc75)
#define PWM43T1 (*(volatile unsigned int __xdata *)0xfc78)
#define PWM43T1H (*(volatile unsigned char __xdata *)0xfc78)
#define PWM43T1L (*(volatile unsigned char __xdata *)0xfc79)
#define PWM43T2 (*(volatile unsigned int __xdata *)0xfc7a)
#define PWM43T2H (*(volatile unsigned char __xdata *)0xfc7a)
#define PWM43T2L (*(volatile unsigned char __xdata *)0xfc7b)
#define PWM43CR (*(volatile unsigned char __xdata *)0xfc7c)
#define PWM43HLD (*(volatile unsigned char __xdata *)0xfc7d)
#define PWM44T1 (*(volatile unsigned int __xdata *)0xfc80)
#define PWM44T1H (*(volatile unsigned char __xdata *)0xfc80)
#define PWM44T1L (*(volatile unsigned char __xdata *)0xfc81)
#define PWM44T2 (*(volatile unsigned int __xdata *)0xfc82)
#define PWM44T2H (*(volatile unsigned char __xdata *)0xfc82)
#define PWM44T2L (*(volatile unsigned char __xdata *)0xfc83)
#define PWM44CR (*(volatile unsigned char __xdata *)0xfc84)
#define PWM44HLD (*(volatile unsigned char __xdata *)0xfc85)
#define PWM45T1 (*(volatile unsigned int __xdata *)0xfc88)
#define PWM45T1H (*(volatile unsigned char __xdata *)0xfc88)
#define PWM45T1L (*(volatile unsigned char __xdata *)0xfc89)
#define PWM45T2 (*(volatile unsigned int __xdata *)0xfc8a)
#define PWM45T2H (*(volatile unsigned char __xdata *)0xfc8a)
#define PWM45T2L (*(volatile unsigned char __xdata *)0xfc8b)
#define PWM45CR (*(volatile unsigned char __xdata *)0xfc8c)
#define PWM45HLD (*(volatile unsigned char __xdata *)0xfc8d)
#define PWM46T1 (*(volatile unsigned int __xdata *)0xfc90)
#define PWM46T1H (*(volatile unsigned char __xdata *)0xfc90)
#define PWM46T1L (*(volatile unsigned char __xdata *)0xfc91)
#define PWM46T2 (*(volatile unsigned int __xdata *)0xfc92)
#define PWM46T2H (*(volatile unsigned char __xdata *)0xfc92)
#define PWM46T2L (*(volatile unsigned char __xdata *)0xfc93)
#define PWM46CR (*(volatile unsigned char __xdata *)0xfc94)
#define PWM46HLD (*(volatile unsigned char __xdata *)0xfc95)
#define PWM47T1 (*(volatile unsigned int __xdata *)0xfc98)
#define PWM47T1H (*(volatile unsigned char __xdata *)0xfc98)
#define PWM47T1L (*(volatile unsigned char __xdata *)0xfc99)
#define PWM47T2 (*(volatile unsigned int __xdata *)0xfc9a)
#define PWM47T2H (*(volatile unsigned char __xdata *)0xfc9a)
#define PWM47T2L (*(volatile unsigned char __xdata *)0xfc9b)
#define PWM47CR (*(volatile unsigned char __xdata *)0xfc9c)
#define PWM47HLD (*(volatile unsigned char __xdata *)0xfc9d)
#define PWM5C (*(volatile unsigned int __xdata *)0xfca0)
#define PWM5CH (*(volatile unsigned char __xdata *)0xfca0)
#define PWM5CL (*(volatile unsigned char __xdata *)0xfca1)
#define PWM5CKS (*(volatile unsigned char __xdata *)0xfca2)
#define PWM5IF (*(volatile unsigned char __xdata *)0xfca5)
#define PWM5FDCR (*(volatile unsigned char __xdata *)0xfca6)
#define PWM50T1 (*(volatile unsigned int __xdata *)0xfcb0)
#define PWM50T1H (*(volatile unsigned char __xdata *)0xfcb0)
#define PWM50T1L (*(volatile unsigned char __xdata *)0xfcb1)
#define PWM50T2 (*(volatile unsigned int __xdata *)0xfcb2)
#define PWM50T2H (*(volatile unsigned char __xdata *)0xfcb2)
#define PWM50T2L (*(volatile unsigned char __xdata *)0xfcb3)
#define PWM50CR (*(volatile unsigned char __xdata *)0xfcb4)
#define PWM50HLD (*(volatile unsigned char __xdata *)0xfcb5)
#define PWM51T1 (*(volatile unsigned int __xdata *)0xfcb8)
#define PWM51T1H (*(volatile unsigned char __xdata *)0xfcb8)
#define PWM51T1L (*(volatile unsigned char __xdata *)0xfcb9)
#define PWM51T2 (*(volatile unsigned int __xdata *)0xfcba)
#define PWM51T2H (*(volatile unsigned char __xdata *)0xfcba)
#define PWM51T2L (*(volatile unsigned char __xdata *)0xfcbb)
#define PWM51CR (*(volatile unsigned char __xdata *)0xfcbc)
#define PWM51HLD (*(volatile unsigned char __xdata *)0xfcbd)
#define PWM52T1 (*(volatile unsigned int __xdata *)0xfcc0)
#define PWM52T1H (*(volatile unsigned char __xdata *)0xfcc0)
#define PWM52T1L (*(volatile unsigned char __xdata *)0xfcc1)
#define PWM52T2 (*(volatile unsigned int __xdata *)0xfcc2)
#define PWM52T2H (*(volatile unsigned char __xdata *)0xfcc2)
#define PWM52T2L (*(volatile unsigned char __xdata *)0xfcc3)
#define PWM52CR (*(volatile unsigned char __xdata *)0xfcc4)
#define PWM52HLD (*(volatile unsigned char __xdata *)0xfcc5)
#define PWM53T1 (*(volatile unsigned int __xdata *)0xfcc8)
#define PWM53T1H (*(volatile unsigned char __xdata *)0xfcc8)
#define PWM53T1L (*(volatile unsigned char __xdata *)0xfcc9)
#define PWM53T2 (*(volatile unsigned int __xdata *)0xfcca)
#define PWM53T2H (*(volatile unsigned char __xdata *)0xfcca)
#define PWM53T2L (*(volatile unsigned char __xdata *)0xfccb)
#define PWM53CR (*(volatile unsigned char __xdata *)0xfccc)
#define PWM53HLD (*(volatile unsigned char __xdata *)0xfccd)
#define PWM54T1 (*(volatile unsigned int __xdata *)0xfcd0)
#define PWM54T1H (*(volatile unsigned char __xdata *)0xfcd0)
#define PWM54T1L (*(volatile unsigned char __xdata *)0xfcd1)
#define PWM54T2 (*(volatile unsigned int __xdata *)0xfcd2)
#define PWM54T2H (*(volatile unsigned char __xdata *)0xfcd2)
#define PWM54T2L (*(volatile unsigned char __xdata *)0xfcd3)
#define PWM54CR (*(volatile unsigned char __xdata *)0xfcd4)
#define PWM54HLD (*(volatile unsigned char __xdata *)0xfcd5)
#define PWM55T1 (*(volatile unsigned int __xdata *)0xfcd8)
#define PWM55T1H (*(volatile unsigned char __xdata *)0xfcd8)
#define PWM55T1L (*(volatile unsigned char __xdata *)0xfcd9)
#define PWM55T2 (*(volatile unsigned int __xdata *)0xfcda)
#define PWM55T2H (*(volatile unsigned char __xdata *)0xfcda)
#define PWM55T2L (*(volatile unsigned char __xdata *)0xfcdb)
#define PWM55CR (*(volatile unsigned char __xdata *)0xfcdc)
#define PWM55HLD (*(volatile unsigned char __xdata *)0xfcdd)
#define PWM56T1 (*(volatile unsigned int __xdata *)0xfce0)
#define PWM56T1H (*(volatile unsigned char __xdata *)0xfce0)
#define PWM56T1L (*(volatile unsigned char __xdata *)0xfce1)
#define PWM56T2 (*(volatile unsigned int __xdata *)0xfce2)
#define PWM56T2H (*(volatile unsigned char __xdata *)0xfce2)
#define PWM56T2L (*(volatile unsigned char __xdata *)0xfce3)
#define PWM56CR (*(volatile unsigned char __xdata *)0xfce4)
#define PWM56HLD (*(volatile unsigned char __xdata *)0xfce5)
#define PWM57T1 (*(volatile unsigned int __xdata *)0xfce8)
#define PWM57T1H (*(volatile unsigned char __xdata *)0xfce8)
#define PWM57T1L (*(volatile unsigned char __xdata *)0xfce9)
#define PWM57T2 (*(volatile unsigned int __xdata *)0xfcea)
#define PWM57T2H (*(volatile unsigned char __xdata *)0xfcea)
#define PWM57T2L (*(volatile unsigned char __xdata *)0xfceb)
#define PWM57CR (*(volatile unsigned char __xdata *)0xfcec)
#define PWM57HLD (*(volatile unsigned char __xdata *)0xfced)
                                    
#define MD3 (*(volatile unsigned char __xdata *)0xfcf0)
#define MD2 (*(volatile unsigned char __xdata *)0xfcf1)
#define MD1 (*(volatile unsigned char __xdata *)0xfcf2)
#define MD0 (*(volatile unsigned char __xdata *)0xfcf3)
#define MD5 (*(volatile unsigned char __xdata *)0xfcf4)
#define MD4 (*(volatile unsigned char __xdata *)0xfcf5)
#define ARCON (*(volatile unsigned char __xdata *)0xfcf6)
#define OPCON (*(volatile unsigned char __xdata *)0xfcf7)

/////////////////////////////////////////////////
//FB00H-FBFFH
/////////////////////////////////////////////////

#define COMEN (*(volatile unsigned char __xdata *)0xfb00)
#define SEGENL (*(volatile unsigned char __xdata *)0xfb01)
#define SEGENH (*(volatile unsigned char __xdata *)0xfb02)
#define LEDCTRL (*(volatile unsigned char __xdata *)0xfb03)
#define LEDCKS (*(volatile unsigned char __xdata *)0xfb04)
#define COM0_DA_L (*(volatile unsigned char __xdata *)0xfb10)
#define COM1_DA_L (*(volatile unsigned char __xdata *)0xfb11)
#define COM2_DA_L (*(volatile unsigned char __xdata *)0xfb12)
#define COM3_DA_L (*(volatile unsigned char __xdata *)0xfb13)
#define COM4_DA_L (*(volatile unsigned char __xdata *)0xfb14)
#define COM5_DA_L (*(volatile unsigned char __xdata *)0xfb15)
#define COM6_DA_L (*(volatile unsigned char __xdata *)0xfb16)
#define COM7_DA_L (*(volatile unsigned char __xdata *)0xfb17)
#define COM0_DA_H (*(volatile unsigned char __xdata *)0xfb18)
#define COM1_DA_H (*(volatile unsigned char __xdata *)0xfb19)
#define COM2_DA_H (*(volatile unsigned char __xdata *)0xfb1a)
#define COM3_DA_H (*(volatile unsigned char __xdata *)0xfb1b)
#define COM4_DA_H (*(volatile unsigned char __xdata *)0xfb1c)
#define COM5_DA_H (*(volatile unsigned char __xdata *)0xfb1d)
#define COM6_DA_H (*(volatile unsigned char __xdata *)0xfb1e)
#define COM7_DA_H (*(volatile unsigned char __xdata *)0xfb1f)
#define COM0_DC_L (*(volatile unsigned char __xdata *)0xfb20)
#define COM1_DC_L (*(volatile unsigned char __xdata *)0xfb21)
#define COM2_DC_L (*(volatile unsigned char __xdata *)0xfb22)
#define COM3_DC_L (*(volatile unsigned char __xdata *)0xfb23)
#define COM4_DC_L (*(volatile unsigned char __xdata *)0xfb24)
#define COM5_DC_L (*(volatile unsigned char __xdata *)0xfb25)
#define COM6_DC_L (*(volatile unsigned char __xdata *)0xfb26)
#define COM7_DC_L (*(volatile unsigned char __xdata *)0xfb27)
#define COM0_DC_H (*(volatile unsigned char __xdata *)0xfb28)
#define COM1_DC_H (*(volatile unsigned char __xdata *)0xfb29)
#define COM2_DC_H (*(volatile unsigned char __xdata *)0xfb2a)
#define COM3_DC_H (*(volatile unsigned char __xdata *)0xfb2b)
#define COM4_DC_H (*(volatile unsigned char __xdata *)0xfb2c)
#define COM5_DC_H (*(volatile unsigned char __xdata *)0xfb2d)
#define COM6_DC_H (*(volatile unsigned char __xdata *)0xfb2e)
#define COM7_DC_H (*(volatile unsigned char __xdata *)0xfb2f)
                                    
#define TSCHEN1 (*(volatile unsigned char __xdata *)0xfb40)
#define TSCHEN2 (*(volatile unsigned char __xdata *)0xfb41)
#define TSCFG1 (*(volatile unsigned char __xdata *)0xfb42)
#define TSCFG2 (*(volatile unsigned char __xdata *)0xfb43)
#define TSWUTC (*(volatile unsigned char __xdata *)0xfb44)
#define TSCTRL (*(volatile unsigned char __xdata *)0xfb45)
#define TSSTA1 (*(volatile unsigned char __xdata *)0xfb46)
#define TSSTA2 (*(volatile unsigned char __xdata *)0xfb47)
#define TSRT (*(volatile unsigned char __xdata *)0xfb48)
#define TSDAT (*(volatile unsigned int __xdata *)0xfb49)
#define TSDATH (*(volatile unsigned char __xdata *)0xfb49)
#define TSDATL (*(volatile unsigned char __xdata *)0xfb4a)
#define TSTH00 (*(volatile unsigned int __xdata *)0xfb50)
#define TSTH00H (*(volatile unsigned char __xdata *)0xfb50)
#define TSTH00L (*(volatile unsigned char __xdata *)0xfb51)
#define TSTH01 (*(volatile unsigned int __xdata *)0xfb52)
#define TSTH01H (*(volatile unsigned char __xdata *)0xfb52)
#define TSTH01L (*(volatile unsigned char __xdata *)0xfb53)
#define TSTH02 (*(volatile unsigned int __xdata *)0xfb54)
#define TSTH02H (*(volatile unsigned char __xdata *)0xfb54)
#define TSTH02L (*(volatile unsigned char __xdata *)0xfb55)
#define TSTH03 (*(volatile unsigned int __xdata *)0xfb56)
#define TSTH03H (*(volatile unsigned char __xdata *)0xfb56)
#define TSTH03L (*(volatile unsigned char __xdata *)0xfb57)
#define TSTH04 (*(volatile unsigned int __xdata *)0xfb58)
#define TSTH04H (*(volatile unsigned char __xdata *)0xfb58)
#define TSTH04L (*(volatile unsigned char __xdata *)0xfb59)
#define TSTH05 (*(volatile unsigned int __xdata *)0xfb5a)
#define TSTH05H (*(volatile unsigned char __xdata *)0xfb5a)
#define TSTH05L (*(volatile unsigned char __xdata *)0xfb5b)
#define TSTH06 (*(volatile unsigned int __xdata *)0xfb5c)
#define TSTH06H (*(volatile unsigned char __xdata *)0xfb5c)
#define TSTH06L (*(volatile unsigned char __xdata *)0xfb5d)
#define TSTH07 (*(volatile unsigned int __xdata *)0xfb5e)
#define TSTH07H (*(volatile unsigned char __xdata *)0xfb5e)
#define TSTH07L (*(volatile unsigned char __xdata *)0xfb5f)
#define TSTH08 (*(volatile unsigned int __xdata *)0xfb60)
#define TSTH08H (*(volatile unsigned char __xdata *)0xfb60)
#define TSTH08L (*(volatile unsigned char __xdata *)0xfb61)
#define TSTH09 (*(volatile unsigned int __xdata *)0xfb62)
#define TSTH09H (*(volatile unsigned char __xdata *)0xfb62)
#define TSTH09L (*(volatile unsigned char __xdata *)0xfb63)
#define TSTH10 (*(volatile unsigned int __xdata *)0xfb64)
#define TSTH10H (*(volatile unsigned char __xdata *)0xfb64)
#define TSTH10L (*(volatile unsigned char __xdata *)0xfb65)
#define TSTH11 (*(volatile unsigned int __xdata *)0xfb66)
#define TSTH11H (*(volatile unsigned char __xdata *)0xfb66)
#define TSTH11L (*(volatile unsigned char __xdata *)0xfb67)
#define TSTH12 (*(volatile unsigned int __xdata *)0xfb68)
#define TSTH12H (*(volatile unsigned char __xdata *)0xfb68)
#define TSTH12L (*(volatile unsigned char __xdata *)0xfb69)
#define TSTH13 (*(volatile unsigned int __xdata *)0xfb6a)
#define TSTH13H (*(volatile unsigned char __xdata *)0xfb6a)
#define TSTH13L (*(volatile unsigned char __xdata *)0xfb6b)
#define TSTH14 (*(volatile unsigned int __xdata *)0xfb6c)
#define TSTH14H (*(volatile unsigned char __xdata *)0xfb6c)
#define TSTH14L (*(volatile unsigned char __xdata *)0xfb6d)
#define TSTH15 (*(volatile unsigned int __xdata *)0xfb6e)
#define TSTH15H (*(volatile unsigned char __xdata *)0xfb6e)
#define TSTH15L (*(volatile unsigned char __xdata *)0xfb6f)

/////////////////////////////////////////////////
//FA00H-FAFFH
/////////////////////////////////////////////////

/////////////////////////////////////////////////
//F900H-F9FFH
/////////////////////////////////////////////////

#define PWMA_ENO2 (*(volatile unsigned char __xdata *)0xf930)
#define PWMA_IOAUX2 (*(volatile unsigned char __xdata *)0xf931)
#define PWMA_CR3 (*(volatile unsigned char __xdata *)0xf932)
#define PWMA_SR3 (*(volatile unsigned char __xdata *)0xf933)
#define PWMA_CCER3 (*(volatile unsigned char __xdata *)0xf934)
#define PWMA_CCMR1X (*(volatile unsigned char __xdata *)0xf938)
#define PWMA_CCMR2X (*(volatile unsigned char __xdata *)0xf939)
#define PWMA_CCMR3X (*(volatile unsigned char __xdata *)0xf93a)
#define PWMA_CCMR4X (*(volatile unsigned char __xdata *)0xf93b)
#define PWMA_CCMR5 (*(volatile unsigned char __xdata *)0xf93c)
#define PWMA_CCMR5X (*(volatile unsigned char __xdata *)0xf93d)
#define PWMA_CCMR6 (*(volatile unsigned char __xdata *)0xf93e)
#define PWMA_CCMR6X (*(volatile unsigned char __xdata *)0xf93f)
#define PWMA_CCR5H (*(volatile unsigned char __xdata *)0xf940)
#define PWMA_CCR5L (*(volatile unsigned char __xdata *)0xf941)
#define PWMA_CCR5X (*(volatile unsigned char __xdata *)0xf942)
#define PWMA_CCR6H (*(volatile unsigned char __xdata *)0xf943)
#define PWMA_CCR6L (*(volatile unsigned char __xdata *)0xf944)

#endif