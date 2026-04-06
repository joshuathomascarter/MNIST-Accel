/* Linker script for RISC-V SoC bare-metal firmware */

MEMORY
{
    ROM (rx)  : ORIGIN = 0x00000000, LENGTH = 8K
    RAM (rwx) : ORIGIN = 0x10000000, LENGTH = 32K
}

SECTIONS
{
    .text : {
        *(.text.start)
        *(.text)
        *(.rodata)
    } > ROM

    .data : {
        *(.data)
    } > ROM AT > ROM  /* Load from ROM, but image goes to RAM if needed */

    .bss : {
        _bss_start = .;
        *(.bss)
        *(COMMON)
        _bss_end = .;
    } > RAM

    /* Define symbols for startup.S */
    _data_start = ADDR(.data);
    _data_end = ADDR(.data) + SIZEOF(.data);
    _data_load = LOADADDR(.data);
}
