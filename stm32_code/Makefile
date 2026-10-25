# STM32 Makefile for Heart-Lung Sound Separation
TARGET = heart_lung_separation
MCU = cortex-m4
FPU = -mfpu=fpv4-sp-d16 -mfloat-abi=hard

CC = arm-none-eabi-gcc
AS = arm-none-eabi-gcc -x assembler-with-cpp
CP = arm-none-eabi-objcopy
SZ = arm-none-eabi-size

CFLAGS = -mcpu=$(MCU) $(FPU) -mthumb -Wall -O2 -std=c99
CFLAGS += -DSTM32F4xx
CFLAGS += -ICore/Inc -IDrivers/STM32F4xx_HAL_Driver/Inc

LDFLAGS = -mcpu=$(MCU) $(FPU) -mthumb -specs=nano.specs -Wl,--gc-sections

# Sources
C_SOURCES = Core/Src/main.c Core/Src/audio_processing.c Core/Src/filter_design.c \
            Core/Src/feature_extract.c Core/Src/classification.c

# Build
all: $(TARGET).elf $(TARGET).bin

$(TARGET).elf: $(C_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(TARGET).bin: $(TARGET).elf
	$(CP) -O binary $< $@

clean:
	rm -f $(TARGET).elf $(TARGET).bin

size: $(TARGET).elf
	$(SZ) $<
