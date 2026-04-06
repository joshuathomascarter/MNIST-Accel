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

INF_SOURCES = startup.S main_inference.c
INF_OBJECTS = startup.o main_inference.o

FULL_INF_SOURCES = startup.S main_inference_full.c
FULL_INF_OBJECTS = startup.o main_inference_full.o

MT_INF_SOURCES = startup.S main_inference_multitile.c
MT_INF_OBJECTS = startup.o main_inference_multitile.o

.PHONY: all clean disasm inference full-inference multitile

all: firmware.hex

inference: firmware_inference.hex

full-inference: firmware_inference_full.hex

multitile: firmware_inference_multitile.hex

%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

firmware.elf: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $@

firmware.hex: firmware.elf
	$(OBJCOPY) -O verilog firmware.elf firmware.hex
	@echo "Generated firmware.hex"

firmware_inference.elf: $(INF_OBJECTS)
	$(LD) $(LDFLAGS) $(INF_OBJECTS) -o $@

firmware_inference.hex: firmware_inference.elf
	$(OBJCOPY) -O verilog firmware_inference.elf firmware_inference.hex
	@echo "Generated firmware_inference.hex"

firmware_inference_full.elf: $(FULL_INF_OBJECTS)
	$(LD) $(LDFLAGS) $(FULL_INF_OBJECTS) -o $@

firmware_inference_full.hex: firmware_inference_full.elf
	$(OBJCOPY) -O verilog firmware_inference_full.elf firmware_inference_full.hex
	@echo "Generated firmware_inference_full.hex"

firmware_inference_multitile.elf: $(MT_INF_OBJECTS)
	$(LD) $(LDFLAGS) $(MT_INF_OBJECTS) -o $@

firmware_inference_multitile.hex: firmware_inference_multitile.elf
	$(OBJCOPY) -O verilog firmware_inference_multitile.elf firmware_inference_multitile.hex
	@echo "Generated firmware_inference_multitile.hex"

clean:
	rm -f $(OBJECTS) $(INF_OBJECTS) $(FULL_INF_OBJECTS) $(MT_INF_OBJECTS) \
	      firmware.elf firmware.hex \
	      firmware_inference.elf firmware_inference.hex \
	      firmware_inference_full.elf firmware_inference_full.hex \
	      firmware_inference_multitile.elf firmware_inference_multitile.hex
.PRECIOUS: firmware.elf
