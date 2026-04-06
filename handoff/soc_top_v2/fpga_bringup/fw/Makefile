# Makefile for RISC-V Bare-Metal Firmware

CROSS_COMPILE ?= riscv64-elf-
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

CFLAGS  = -march=rv32i_zicsr -mabi=ilp32 -nostdlib -ffreestanding -O2 -I.
ASFLAGS = -march=rv32i_zicsr -mabi=ilp32
LDFLAGS = -T link.ld -nostdlib -m elf32lriscv

SOURCES = startup.S main.c
OBJECTS = startup.o main.o

.PHONY: all clean disasm

all: firmware.hex

%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

firmware.elf: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $@

firmware.hex: firmware.elf
	$(OBJCOPY) -O verilog firmware.elf firmware.hex
	@echo "Generated firmware.hex"

disasm: firmware.elf
	$(OBJDUMP) -d firmware.elf | head -50
	@echo "..."

clean:
	rm -f $(OBJECTS) firmware.elf firmware.hex

.PRECIOUS: firmware.elf
