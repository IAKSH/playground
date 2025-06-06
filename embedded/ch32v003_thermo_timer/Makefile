TOOL_CHAIN_PATH ?= "/home/lain/MRS_Toolchain/RISC-V Embedded GCC12/bin"
OPENOCD_PATH	?= /home/lain/MRS_Toolchain/OpenOCD/OpenOCD/bin
PROJECT_NAME    = app

TOP_DIR     := .
OUTPUT_DIR  := $(TOP_DIR)/Build
STARTUP_DIR := $(TOP_DIR)/Startup
CORE_DIR    := $(TOP_DIR)/Core
DEBUG_DIR   := $(TOP_DIR)/Debug
SPL_DIR     := $(TOP_DIR)/Peripheral
USER_DIR    := $(TOP_DIR)/User

LD_FILE     := $(TOP_DIR)/Ld/Link.ld
OPENOCD_CFG := $(TOP_DIR)/Util/wch-riscv.cfg
MAP_FILE    := $(OUTPUT_DIR)/$(PROJECT_NAME).map
ELF_FILE    := $(OUTPUT_DIR)/$(PROJECT_NAME).elf
HEX_FILE    := $(OUTPUT_DIR)/$(PROJECT_NAME).hex
LST_FILE    := $(OUTPUT_DIR)/$(PROJECT_NAME).lst
SIZ_FILE    := $(OUTPUT_DIR)/$(PROJECT_NAME).siz

INCLUDES := $(INCLUDES)
INCLUDES += -I $(TOP_DIR)/User
INCLUDES += -I $(TOP_DIR)/Core
INCLUDES += -I $(TOP_DIR)/Debug
INCLUDES += -I $(TOP_DIR)/Peripheral/inc

CCFLAGS := -march=rv32ecxw \
           -mabi=ilp32e \
           -msmall-data-limit=0 \
           -msave-restore \
           -Os \
           -fmessage-length=0 \
           -fsigned-char \
           -ffunction-sections \
           -fdata-sections \
           -fno-common \
           -Wunused -Wuninitialized -g

all: $(HEX_FILE) $(LST_FILE) $(SIZ_FILE)

STARTUP_SRCS := $(wildcard $(STARTUP_DIR)/*.S)
STARTUP_OBJS := $(patsubst $(STARTUP_DIR)/%.S, $(OUTPUT_DIR)/startup/%.o, $(STARTUP_SRCS)) # patsubst: in #3 replace #1 to #2

CORE_SRCS := $(wildcard $(CORE_DIR)/*.c)
CORE_OBJS := $(patsubst $(CORE_DIR)/%.c, $(OUTPUT_DIR)/core/%.o, $(CORE_SRCS))

DEBUG_SRCS := $(wildcard $(DEBUG_DIR)/*.c)
DEBUG_OBJS := $(patsubst $(DEBUG_DIR)/%.c, $(OUTPUT_DIR)/debug/%.o, $(DEBUG_SRCS))

SPL_SRCS := $(wildcard $(SPL_DIR)/src/*.c)
SPL_OBJS := $(patsubst $(SPL_DIR)/src/%.c, $(OUTPUT_DIR)/spl/%.o, $(SPL_SRCS))

USER_SRCS := $(wildcard $(USER_DIR)/*.c)
USER_OBJS := $(patsubst $(USER_DIR)/%.c, $(OUTPUT_DIR)/user/%.o, $(USER_SRCS))

USER_DRIVER_SRCS := $(wildcard $(USER_DIR)/driver/*.c)
USER_DRIVER_OBJS := $(patsubst $(USER_DIR)/driver/%.c, $(OUTPUT_DIR)/user/driver/%.o, $(USER_DRIVER_SRCS))

$(OUTPUT_DIR)/startup/%.o: $(STARTUP_DIR)/%.S
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) -x assembler -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(OUTPUT_DIR)/core/%.o: $(CORE_DIR)/%.c
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) $(INCLUDES) -std=gnu99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(OUTPUT_DIR)/debug/%.o: $(DEBUG_DIR)/%.c
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) $(INCLUDES) -std=gnu99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(OUTPUT_DIR)/spl/%.o: $(SPL_DIR)/src/%.c
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) $(INCLUDES) -std=gnu99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(OUTPUT_DIR)/user/%.o: $(USER_DIR)/%.c
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) $(INCLUDES) -std=gnu99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(OUTPUT_DIR)/user/driver/%.o: $(USER_DIR)/driver/%.c
	@mkdir -p $(@D)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) $(INCLUDES) -std=gnu99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"

$(ELF_FILE): $(STARTUP_OBJS) $(CORE_OBJS) $(DEBUG_OBJS) $(SPL_OBJS) $(USER_OBJS) $(USER_DRIVER_OBJS)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-gcc $(CCFLAGS) -T $(LD_FILE) -nostartfiles -Xlinker --gc-sections -Wl,-Map,$(MAP_FILE) --specs=nano.specs --specs=nosys.specs -o $(ELF_FILE) $(USER_DRIVER_OBJS) $(USER_OBJS) $(STARTUP_OBJS) $(CORE_OBJS) $(DEBUG_OBJS) $(SPL_OBJS)

$(HEX_FILE): $(ELF_FILE)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-objcopy -O ihex $(ELF_FILE) $(HEX_FILE)

$(LST_FILE): $(ELF_FILE)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-objdump --all-headers --demangle --disassemble $(ELF_FILE) > $(LST_FILE)

$(SIZ_FILE): $(ELF_FILE)
	$(TOOL_CHAIN_PATH)/riscv-wch-elf-size --format=berkeley $(ELF_FILE)


.PHONY: clean
clean:
	rm -f $(OUTPUT_DIR)/startup/*
	rm -f $(OUTPUT_DIR)/core/*
	rm -f $(OUTPUT_DIR)/debug/*
	rm -f $(OUTPUT_DIR)/spl/*
	rm -rf $(OUTPUT_DIR)/user/*
	rm -f $(OUTPUT_DIR)/*.*

erase:
	$(OPENOCD_PATH)/openocd -f $(OPENOCD_CFG) -c init -c halt -c "flash erase_sector wch_riscv 0 last" -c exit

# flash:
# 	$(OPENOCD_PATH)/openocd -f $(OPENOCD_CFG) -c init -c halt -c "flash erase_sector wch_riscv 0 last " -c "program $(ELF_FILE)" -c "verify_image $(ELF_FILE)" -c reset -c resume -c exit

flash:
	$(OPENOCD_PATH)/openocd -f $(OPENOCD_CFG) -c init -c halt -c "flash erase_sector wch_riscv 0 last " -c "program $(ELF_FILE)" -c reset -c resume -c exit


reset:
	$(OPENOCD_PATH)/openocd -f $(OPENOCD_CFG) -c init -c reset -c resume -c exit

debug:
	$(OPENOCD_PATH)/openocd -f $(OPENOCD_CFG)