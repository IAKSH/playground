#include "stc8g.h"
#include "oled.h"
#include "i2c.h"

#define INA226_ADDR 0x40         // INA226 7-bit I2C address (default, adjust based on your wiring)
#define INA226_CALIBRATION 4096  // Calibration value for INA226 (adjust according to shunt resistor and current range)

float bus_voltage = 0;
float current = 0;
float power = 0;
unsigned char idle_timer = 0;
bit is_oled_on = 1;

// Delay of approximately 1000ms for an 11.0592MHz clock
void Delay1000ms(void) {
    unsigned char i, j, k;
    i = 57;
    j = 27;
    k = 112;
    do {
        do {
            while (--k);
        } while (--j);
    } while (--i);
}

// Initialize INA226 by writing the configuration and calibration registers
void INA226_Init(void)
{
    // Write configuration register (address 0x00)
    // Example configuration: 0x4127 (averaging, conversion time and operating mode settings)
    unsigned char config_high = 0x41;
    unsigned char config_low  = 0x27;
	
    unsigned char calib_high = INA226_CALIBRATION >> 8;  // High byte of calibration value
    unsigned char calib_low  = INA226_CALIBRATION & 0xFF; // Low byte of calibration value

    Start();
    Send_Data(INA226_ADDR << 1 | 0);  // Write mode
    RecvACK();
    Send_Data(0x00);                  // Pointer to configuration register
    RecvACK();
    Send_Data(config_high);           // High byte of configuration
    RecvACK();
    Send_Data(config_low);            // Low byte of configuration
    RecvACK();
    Stop();

		// Write calibration register (address 0x05)
    Start();
    Send_Data(INA226_ADDR << 1 | 0);  // Write mode
    RecvACK();
    Send_Data(0x05);                  // Pointer to calibration register
    RecvACK();
    Send_Data(calib_high);            // High byte of calibration value
    RecvACK();
    Send_Data(calib_low);             // Low byte of calibration value
    RecvACK();
    Stop();
}

// Read the INA226 bus voltage from register 0x02 and convert the raw value to volts
float INA226_ReadBusVoltage(void)
{
    unsigned char high, low;
    unsigned int raw;
    float voltage_value;

    Start();
    Send_Data(INA226_ADDR << 1 | 0);  // Write mode
    RecvACK();
    Send_Data(0x02);                  // Pointer to bus voltage register
    RecvACK();
    Start();
    Send_Data(INA226_ADDR << 1 | 1);  // Read mode
    RecvACK();
    high = Recv_Data();
    SendACK();
    low = Recv_Data();
    SendNAK();
    Stop();

    raw = ((unsigned int)high << 8) | low;
    // Each LSB equals 1.25 mV (0.00125 V per bit)
    voltage_value = raw * 0.00125;
    return voltage_value;
}

// Read the INA226 current from register 0x04 and convert the raw value to amperes
float INA226_ReadCurrent(void)
{
    unsigned char high, low;
    signed int raw;
    float current_value;

    Start();
    Send_Data(INA226_ADDR << 1 | 0);  // Write mode
    RecvACK();
    Send_Data(0x04);                  // Pointer to current register
    RecvACK();
    Start();
    Send_Data(INA226_ADDR << 1 | 1);  // Read mode
    RecvACK();
    high = Recv_Data();
    SendACK();
    low = Recv_Data();
    SendNAK();
    Stop();

    // Combine the two bytes into a signed 16-bit value
    raw = (signed int)((high << 8) | low);
    // Assuming 1 mA per bit (0.001 A), convert raw value to amperes
    current_value = raw * 0.001;
    return current_value;
}

// Read the INA226 power from register 0x03 and convert the raw value to watts
float INA226_ReadPower(void)
{
    unsigned char high, low;
    unsigned int raw;
    float power_value;

    Start();
    Send_Data(INA226_ADDR << 1 | 0);  // Write mode
    RecvACK();
    Send_Data(0x03);                  // Pointer to power register
    RecvACK();
    Start();
    Send_Data(INA226_ADDR << 1 | 1);  // Read mode
    RecvACK();
    high = Recv_Data();
    SendACK();
    low = Recv_Data();
    SendNAK();
    Stop();

    raw = ((unsigned int)high << 8) | low;
    // Assuming power LSB = 25 * current LSB (with current LSB = 0.001 A) gives 0.025 watt per bit
    power_value = raw * 0.025;
    return power_value;
}

// External Interrupt 1 service routine triggered by the INA226 Alert signal on P3.2
void External_Interrupt_1(void) interrupt 2 {
    if (!is_oled_on) {
        OLED_Init();
        OLED_Clear();
        is_oled_on = 1;
    }
    idle_timer = 0;   // Reset idle timer upon receiving an alert signal
    PCON &= 0x7F;     // Exit low power mode
}

void main(void) {
    // Configure port modes: set P3 and P5 as inputs
    P3M0 = 0x00;
    P3M1 = 0x00;
    P5M0 = 0x00;
    P5M1 = 0x00;

    // Set P3.2 as input for the Alert signal from INA226
    P3M0 |= 0x04;
    P3M1 |= 0x04;

    // Configure I2C2 pins through the P_SW2 register
    P_SW2 = 0x80;
    P_SW2 &= ~(1 << 5);
    P_SW2 |= (1 << 4);

    I2CCFG = 0xe0;
    I2CMSST = 0x00;
    
    // Initialize OLED display and INA226 sensor
    OLED_Init();
    OLED_Clear();
    INA226_Init();

    // Enable interrupts: Global, External Interrupt 1 (falling edge trigger) for alert handling
    EA = 1;
    EX1 = 1;
    IT1 = 1;

    OLED_ShowString(16, 1, "Hello world!", 16);
    Delay1000ms();
    OLED_Clear();

    // Main loop: Read and display values from INA226 every second
    while (1) {
        if (is_oled_on) {
            // Read measurements from INA226
            bus_voltage = INA226_ReadBusVoltage();
            current = INA226_ReadCurrent();
            power = INA226_ReadPower();

            // Display bus voltage on OLED
					  OLED_ShowString(0, 0, "V:", 16);
            OLED_ShowFloat(20, 0, bus_voltage, 2, 16);

            // Display current on OLED
					  OLED_ShowString(70, 0, "A:", 16);
            OLED_ShowFloat(90, 0, current, 3, 16);

            // Display power on OLED
            OLED_ShowString(0, 2, "W:", 16);
            OLED_ShowFloat(20, 2, power, 3, 16);
					
            OLED_ShowString(70, 2, "%:", 16);
            OLED_ShowFloat(90, 2, bus_voltage / 12.0f * 100.0f, 3, 16);

            // If idle for 30 seconds, clear OLED and enter low power mode
            if (++idle_timer > 30) {
                OLED_Clear();
                is_oled_on = 0;
                PCON |= 0x02;  // Enter low power mode
            }
        }
        Delay1000ms();
    }
}
