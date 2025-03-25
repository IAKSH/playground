# Makefile for SDCC (mcs51) project with build directory

CC = sdcc
PACKIHX = packihx
CFLAGS = -mmcs51 -Isrc
BUILD_DIR = build

SRCS = src/core/main.c \
		   src/drivers/i2c/i2c.c \
		   src/drivers/ina226/ina226.c \
		   src/drivers/ssd1306/ssd1306_fonts.c \
		   src/drivers/ssd1306/ssd1306.c \
		   src/drivers/uart/uart.c \
		   src/utils/delay/soft_delay.c

OBJS = $(patsubst src/%, $(BUILD_DIR)/%, $(SRCS:.c=.rel))

TARGET = $(BUILD_DIR)/main.ihx
TARGET_HEX = $(BUILD_DIR)/main.hex

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)
	$(PACKIHX) $(TARGET) > $(TARGET_HEX)

$(BUILD_DIR)/%.rel: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
