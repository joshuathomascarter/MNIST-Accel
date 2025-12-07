t o # Josh Carter
**Hardware Engineer | FPGA Specialist | Open Source Contributor**
Montreal, QC | [GitHub Link] | [LinkedIn Link] | [Email]

---

## **SUMMARY**
"Scrappy" Hardware Engineer with deep experience in custom RTL design, AXI4 interconnects, and AI Acceleration. Proven track record of building full-stack FPGA systems from scratch (RTL to Python Driver). Winner of "Best Hardware Hack" at ConUHacks 2026.

---

## **AWARDS & HONORS**
*   **Winner, Best Hardware Hack** – *ConUHacks 2026*
    *   Built "Sky-Eye": A real-time drone gesture control system using a custom FPGA accelerator for hand tracking.
*   **Global Finalist** – *Digilent Design Contest 2026*
    *   Selected as top 10 worldwide for the "ACCEL-v1" Sparse AI Accelerator project.

---

## **TECHNICAL EXPERIENCE**

### **Open Source Contributor**
*Spring 2026*
*   **Cocotb:** Contributed fixes to the AXI4 Monitor extension, improving burst transaction logging.
*   **Verilator:** Optimized simulation runtime for sparse memory models.

---

## **ENGINEERING PROJECTS**

### **ACCEL-v2: RISC-V SoC with Vector AI Extension**
*Jan 2026 – May 2026*
*   **Integration:** Integrated the ACCEL-v1 Systolic Array as a custom **Vector Coprocessor (ROCC)** attached to a **RISC-V (Ibex)** core.
*   **System Design:** Implemented a custom **Network-on-Chip (NoC)** router to arbitrate traffic between the CPU, Accelerator, and HDMI Video Output.
*   **Software:** Wrote a **Linux Kernel Module (LKM)** to expose the accelerator as a character device (`/dev/accel0`), enabling standard userspace applications to offload matrix math.
*   **Application:** Demonstrated real-time **BERT-Tiny** inference (Transformer model) running entirely on the FPGA fabric.

### **ACCEL-v1: Sparse Matrix AI Accelerator**
*Nov 2025 – Dec 2025*
*   **Architecture:** Designed an 8x8 Systolic Array for **Sparse Matrix Multiplication**, utilizing BSR format to skip zero-ops.
*   **Performance:** Achieved **50 GOPS (Effective)** on Zynq-7020, outperforming commercial HLS solutions by 5x in efficiency.
*   **Data Movement:** Built a custom **AXI4 DMA Engine** with scatter-gather support and a 2-to-1 Read Arbiter for maximizing DDR bandwidth.
*   **Impact:** Featured project in the Digilent Design Contest; open-sourced on GitHub with 100+ stars.

---

## **SKILLS**
*   **Hardware:** SystemVerilog, RISC-V Architecture, AXI4/TileLink, NoC Design, DSP Optimization.
*   **Verification:** UVM, Cocotb, Formal Verification (SymbiYosys).
*   **Software:** C/C++ (Embedded), Python, Linux Kernel Dev, PyTorch.
*   **Tools:** Vivado, Verilator, GTKWave, Yosys, Git/GitHub Actions.

---

## **EDUCATION**
**Concordia University**
*Bachelor of Engineering, Computer Engineering*
*   *Activities:* Lead FPGA Engineer @ Space Concordia (Satellite Team).
