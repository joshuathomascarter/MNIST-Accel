// RISC-V Bare-Metal Startup Code
// Boot sequence: initialize stack, BSS, call main

    .section .text.start
    .align 2
    .globl _start

_start:
    // Initialize stack pointer (top of SRAM: 0x10008000)
    li sp, 0x10008000
    
    // Initialize exception handling (set mtvec)
    la t0, trap_handler
    csrw mtvec, t0
    
    // Clear BSS section (if any)
    la t0, _bss_start
    la t1, _bss_end
bss_loop:
    bge t0, t1, bss_done
    sw zero, 0(t0)
    addi t0, t0, 4
    jal zero, bss_loop
    
bss_done:
    // Copy .data section from ROM to RAM (if used)
    // For now, skip - we'll put everything in ROM
    
    // Enable interrupts (set MIE, MEIE, MTIE)
    li t0, 0x888  // mie = MIE | MEIE | MTIE
    csrw mie, t0
    
    // Set M-mode default ISR  
    li t0, 0x1800  // mstatus MIE bit
    csrrs zero, mstatus, t0
    
    // Call main
    call main
    
    // If main returns, enter infinite loop (ebreak)
    ebreak
    wfi
    jal zero, _start

// Simple trap handler
trap_handler:
    csrr t0, mcause
    
    // Check for timer interrupt (mcause = 0x80000007 for M-mode timer)
    li t1, 0x80000007
    beq t0, t1, handle_timer
    
    // Check for external interrupt (mcause = 0x8000000B)
    li t1, 0x8000000B
    beq t0, t1, handle_external
    
    jal zero, trap_end
    
handle_timer:
    // Timer interrupt
    li t0, 0x2000000C  // MTIMECMP register at 32-bit boundary
    li t1, 0xFFFFFFFF
    sw t1, 0(t0)  // Set next timeout far in future
    li t1, 0xFFFFFFFF
    sw t1, 4(t0)
    
    // Toggle LED (GPIO bit 0)
    li t0, 0x20020004  // GPIO OUT register
    lw t1, 0(t0)
    xori t1, t1, 1
    sw t1, 0(t0)
    
    jal zero, trap_end
    
handle_external:
    // External interrupt (UART or other)
    jal zero, trap_end
    
trap_end:
    // Restore registers and return from trap
    mret
