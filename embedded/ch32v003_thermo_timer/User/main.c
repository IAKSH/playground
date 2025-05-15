/********************************** (C) COPYRIGHT *******************************
 * File Name          : main.c
 * Author             : WCH
 * Version            : V1.0.0
 * Date               : 2023/12/25
 * Description        : Main program body.
 *********************************************************************************
 * Copyright (c) 2021 Nanjing Qinheng Microelectronics Co., Ltd.
 * Attention: This software (modified or not) and binary are used for 
 * microcontroller manufactured by Nanjing Qinheng Microelectronics.
 *******************************************************************************/

/*
 *@Note
 *Multiprocessor communication mode routine:
 *Master:USART1_Tx(PD5)\USART1_Rx(PD6).
 *This routine demonstrates that USART1 receives the data sent by CH341 and inverts
 *it and sends it (baud rate 115200).
 *
 *Hardware connection:PD5 -- Rx
 *                     PD6 -- Tx
 *
 */

#include "debug.h"
#include "i2c_interface.h"
#include "ssd1306.h"
#include "ssd1306_fonts.h"
#include "ssd1306_tests.h"
#include "aht20_basic.h"

/* Global define */


/* Global Variable */
vu8 val;

/*********************************************************************
 * @fn      USARTx_CFG
 *
 * @brief   Initializes the USART2 & USART3 peripheral.
 *
 * @return  none
 */
void USARTx_CFG(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure = {0};
    USART_InitTypeDef USART_InitStructure = {0};

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOD | RCC_APB2Periph_USART1, ENABLE);

    /* USART1 TX-->D.5   RX-->D.6 */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_30MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOD, &GPIO_InitStructure);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOD, &GPIO_InitStructure);

    USART_InitStructure.USART_BaudRate = 115200;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;

    USART_Init(USART1, &USART_InitStructure);
    USART_Cmd(USART1, ENABLE);
}

void test_led_init(void) {
    GPIO_InitTypeDef GPIO_InitStructure = {0};

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_30MHz;
    GPIO_Init(GPIOC, &GPIO_InitStructure);
}

void led_error_loop(uint16_t ms) {
    uint8_t i = 0;
    while(1) {
        Delay_Ms(ms);
        GPIO_WriteBit(GPIOC, GPIO_Pin_4, (i == 0) ? (i = Bit_SET) : (i = Bit_RESET));
    }
}

void adc_init(void) {
    ADC_InitTypeDef adc_init_structure = {0};
    GPIO_InitTypeDef gpio_init_structure = {0};

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1,ENABLE);
    RCC_ADCCLKConfig(RCC_PCLK2_Div8);

    gpio_init_structure.GPIO_Pin = GPIO_Pin_2;
    gpio_init_structure.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_Init(GPIOA,&gpio_init_structure);

    ADC_DeInit(ADC1);
    adc_init_structure.ADC_Mode = ADC_Mode_Independent;
    adc_init_structure.ADC_ScanConvMode = DISABLE;
    adc_init_structure.ADC_ContinuousConvMode = DISABLE;
    adc_init_structure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
    adc_init_structure.ADC_DataAlign = ADC_DataAlign_Right;
    adc_init_structure.ADC_NbrOfChannel = 1;
    ADC_Init(ADC1,&adc_init_structure);

    ADC_RegularChannelConfig(ADC1,ADC_Channel_0,1,ADC_SampleTime_241Cycles);
    ADC_Calibration_Vol(ADC1,ADC_CALVOL_50PERCENT);
    ADC_Cmd(ADC1,ENABLE);

    ADC_ResetCalibration(ADC1);
    while(ADC_GetResetCalibrationStatus(ADC1));
    ADC_StartCalibration(ADC1);
    while(ADC_GetCalibrationStatus(ADC1));
}

#define ADC_SAMPLE_COUNT 10

float read_adc_voltage(void) {
    uint32_t adc_sum = 0;
    for (int i = 0; i < ADC_SAMPLE_COUNT; i++) {
        ADC_SoftwareStartConvCmd(ADC1, ENABLE);
        while (!ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC));
        adc_sum += ADC_GetConversionValue(ADC1);
    }
    float adc_avg = (float)adc_sum / ADC_SAMPLE_COUNT;
    return (adc_avg * 5.0f / 1024.0f); // 5V, 10-bit resolution
}


/*********************************************************************
 * @fn      main
 *
 * @brief   Main program.
 *
 * @return  none
 */
int main(void) {
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_1);
    SystemCoreClockUpdate();
    systick_init();
#if (SDI_PRINT == SDI_PR_OPEN)
    SDI_Printf_Enable();
#else
    USART_Printf_Init(115200);
#endif
    printf("SystemClk:%d\r\n",SystemCoreClock);
    printf( "ChipID:%08x\r\n", DBGMCU_GetCHIPID() );
    USARTx_CFG();

    test_led_init();
    adc_init();
    i2c_setup();

    char buf[32];

    ssd1306_Init();
    ssd1306_Fill(Black);

    ssd1306_SetCursor(0,0);
    snprintf(buf,sizeof(buf),"SystemClk: %dMHz",SystemCoreClock / 1000000);
    ssd1306_WriteString(buf,Font_6x8,White);
    ssd1306_UpdateScreen();

    if(aht20_basic_init() != 0) {
        ssd1306_Fill(White);
        ssd1306_SetCursor(12,28);
        ssd1306_WriteString("AHT20 init failed!",Font_6x8,Black);
        ssd1306_UpdateScreen();
        led_error_loop(50);
    }
    else {
        ssd1306_SetCursor(0,12);
        ssd1306_WriteString("AHT20 ok!",Font_6x8,White);
        ssd1306_UpdateScreen();
    }

    float temp;
    uint8_t humi;
    float adc_volt;

    while(1) {
        if(aht20_basic_read((float*)&temp,(uint8_t*)&humi) != 0) {
            ssd1306_Fill(White);
            ssd1306_SetCursor(16,28);
            ssd1306_WriteString("AHT20 Dead!",Font_6x8,Black);
            ssd1306_UpdateScreen();
            aht20_basic_deinit();
            led_error_loop(1000);
        }
        else {
            ssd1306_SetCursor(0,24);
            snprintf(buf,sizeof(buf),"temp: %d.%d",(int)temp,(int)(temp * 100) % 100);
            ssd1306_WriteString(buf,Font_6x8,White);

            ssd1306_SetCursor(0,36);
            snprintf(buf,sizeof(buf),"humi: %d",(int)humi);
            ssd1306_WriteString(buf,Font_6x8,White);

            ssd1306_SetCursor(0,48);
            // VDD=5v, 10bit ADC
            adc_volt = read_adc_voltage();
            snprintf(buf,sizeof(buf),"volt: %d.%d",(int)adc_volt,(int)(adc_volt * 100) % 100);
            ssd1306_WriteString(buf,Font_6x8,White);

            ssd1306_UpdateScreen();
        }
    }
}
