#include <stdint.h>
#include <drivers/i2c/i2c.h>
#include "ina226.h"
#include "ina226_config.h"

void ina226_init(void) {
    // Write configuration register (address 0x00)
    // Example configuration: 0x4127 (averaging, conversion time and operating mode settings)
    uint8_t config_high = 0x41;
    uint8_t config_low  = 0x27;
	
    uint8_t calib_high = INA226_CALIBRATION >> 8;  // High byte of calibration value
    uint8_t calib_low  = INA226_CALIBRATION & 0xFF; // Low byte of calibration value

    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 0);  // Write mode
    i2c_recv_ack();
    i2c_send_data(0x00);                  // Pointer to configuration register
    i2c_recv_ack();
    i2c_send_data(config_high);           // High byte of configuration
    i2c_recv_ack();
    i2c_send_data(config_low);            // Low byte of configuration
    i2c_recv_ack();
    i2c_stop();

	// Write calibration register (address 0x05)
    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 0);  // Write mode
    i2c_recv_ack();
    i2c_send_data(0x05);                  // Pointer to calibration register
    i2c_recv_ack();
    i2c_send_data(calib_high);            // High byte of calibration value
    i2c_recv_ack();
    i2c_send_data(calib_low);             // Low byte of calibration value
    i2c_recv_ack();
    i2c_stop();
}

float ina226_read_volt(void) {
    uint8_t high, low;
    uint16_t raw;
    float voltage_value;

    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 0);  // Write mode
    i2c_recv_ack();
    i2c_send_data(0x02);                  // Pointer to bus voltage register
    i2c_recv_ack();
    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 1);  // Read mode
    i2c_recv_ack();
    high = i2c_recv_data();
    i2c_send_ack();
    low = i2c_recv_data();
    i2c_send_nak();
    i2c_stop();

    raw = ((uint16_t)high << 8) | low;
    // Each LSB equals 1.25 mV (0.00125 V per bit)
    voltage_value = raw * 0.00125;
    return voltage_value;
}

float ina226_read_current(void) {
    uint8_t high, low;
    signed int raw;
    float current_value;

    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 0);  // Write mode
    i2c_recv_ack();
    i2c_send_data(0x04);                  // Pointer to current register
    i2c_recv_ack();
    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 1);  // Read mode
    i2c_recv_ack();
    high = i2c_recv_data();
    i2c_send_ack();
    low = i2c_recv_data();
    i2c_send_nak();
    i2c_stop();

    // Combine the two bytes into a signed 16-bit value
    raw = (signed int)((high << 8) | low);
    // Assuming 1 mA per bit (0.001 A), convert raw value to amperes
    current_value = raw * 0.001;
    return current_value;
}

float ina226_read_power(void) {
    uint8_t high, low;
    uint16_t raw;
    float power_value;

    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 0);  // Write mode
    i2c_recv_ack();
    i2c_send_data(0x03);                  // Pointer to power register
    i2c_recv_ack();
    i2c_start();
    i2c_send_data(INA226_ADDR << 1 | 1);  // Read mode
    i2c_recv_ack();
    high = i2c_recv_data();
    i2c_send_ack();
    low = i2c_recv_data();
    i2c_send_nak();
    i2c_stop();

    raw = ((uint16_t)high << 8) | low;
    // Assuming power LSB = 25 * current LSB (with current LSB = 0.001 A) gives 0.025 watt per bit
    power_value = raw * 0.025;
    return power_value;
}