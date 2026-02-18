# Networking & Outreach Strategy (Feb-Apr 2026)

## Document Overview
**Goal:** Leverage your "MNIST Accelerator with BSR Sparsity" project to land a Research Assistant (RA) position or industry role in hardware/memory architecture.
**Location:** Montreal, QC
**Project Relevance:** 
- **SystemVerilog/FPGA:** Demonstrates RTL design skills.
- **BSR Sparsity:** Shows understanding of memory bandwidth constraints.
- **AXI4/DMA:** Proves you can build real system interfaces.
- **Roofline Analysis:** Indicates performance engineering mindset.

---

## 📅 Events Calendar (Feb-April 2026)

| Event | Date | Relevance | Action Item |
|-------|------|-----------|-------------|
| **ConFoo Montreal** | Feb 25-27, 2026 | Has embedded/AI tracks. Good for networking with tech leads. | Attend "Embedded Systems" or "AI Performance" talks. Bring your board (Pynq-Z2) if allowed. |
| **Beyond Matrix Multiply** | Mar 5, 2026 | **High.** Talk focuses on SIMD/SIMT architectures beyond basic matmul. | Talk to the speaker about your systolic array BSR implementation. |
| **IEEE Montreal Events** | Check Website | Technical talks often hosted at Concordia/McGill. | Look for "Circuits and Systems Society" (CASS) events. |
| **Capstone Showcases** | Early April | **Critical.** Engineering final year exhibitions (Concordia/McGill). | Visit to see who is sponsoring (companies often send recruiters/engineers). |
| **PyCon CA** | Late April | Python for hardware verification (cocotb) is a growing niche. | If you use cocotb (you do!), this is a talking point. |

**Hackathons (To Watch):**
- **McHacks (McGill):** usually Jan/Feb. If passed, check for post-event challenges.
- **ConUHacks (Concordia):** usually Jan/Feb. Check results to see if hardware projects won (rare, makes you stand out).

---

## 🎓 Professors & Labs (Academic Research)

**Subject:** RA Inquiry - Sparse Matrix Accelerator on FPGA (BSR/Systolic)

| University | Professor / Lab | Research Focus | Specific Connection to Your Project |
|------------|-----------------|----------------|-------------------------------------|
| **Concordia** | **Dr. Sofiène Tahar** (Hardware Verification Group) | Formal Verification, VLSI, System-on-Chip. | "I've implemented a formal verification testbench for my AXI4 arbiter and used functional coverage." |
| **McGill** | **Dr. Warren Gross** (ISP Lab) | Signal Processing, AI Hardware, Stochastic Computing. | "My project implements integer quantization (INT8) similar to the low-precision architectures your lab investigates." |
| **McGill** | **Dr. Zeljko Zilic** | Embedded Systems, Debug/Test, Interconnects. | "I built a custom performance monitor with AXI-Lite readback to profile stalls, relevant to on-chip interconnect research." |
| **PolyMtl** | **Dr. Yvon Savaria** | Microelectronics, High-speed interconnects. | "I optimized memory bandwidth using Block Sparse Row (BSR) format to reduce off-chip memory access." |
| **ETS** | **Dr. Jean-François Boland** | Avionics, Safety-critical hardware. | "My design uses rigorous SystemVerilog assertions, relevant for safety-critical hardware verification." |

**Action:** Email Dr. Tahar *first* as you intended, but have drafts ready for Gross and Zilic if you don't hear back in 1 week.

---

## 🏢 Industry Targets (Montreal)

**Pitch:** "I am an FPGA engineer who understands **Computer Architecture** and **Memory Bottlenecks**, not just a Verilog coder."

| Company | Role Focus | Why You Fit | Project Hook |
|---------|------------|-------------|--------------|
| **Matrox** | Video/Imaging | Real-time stream processing on FPGA. | "My systolic array processes streaming data (AXI-Stream) just like a video pipeline." |
| **Genetec** | Security/IoT | Custom hardware appliances, video analytics. | "I implemented a complete driver stack (C++ to RTL) for an edge AI accelerator." |
| **Thales / CAE** | Avionics/Sim | Low-latency hardware, reliable systems. | "My double-buffered memory architecture ensures deterministic latency, critical for simulation/avionics." |
| **Rambus (Hardent)** | IP Cores | Display compression, Memory controllers. | "I wrote a custom DMA controller and understand AXI burst transactions and memory alignment." |
| **CMC Microsystems** | R&D Support | Enabling chip research in Canada. | "I have end-to-end flow experience (RTL to Pynq deployment) useful for helping researchers prototype." |
| **Deeplite** | AI Optimization | Software/Hardware co-design for efficiency. | "I built a hardware-aware BSR encoder in C++ that matches the RTL's specific sparsity constraints." |

---

## 📝 Email Template for Dr. Tahar

**Subject:** RA Position Inquiry - Built Sparse Matrix FPGA Accelerator (AXI/Int8)

**Body:**
Dear Professor Tahar,

I am a Computer Engineering student broadly interested in **Hardware Verification and Computer Architecture**. I am writing to express my strong interest in joining your lab as a Research Assistant.

I recently completed a custom **FPGA-based Neural Network Accelerator** (deployed on Pynq-Z2) which focuses on memory-efficiency. Key technical features include:
*   **14x14 Systolic Array** with Int8 quantization.
*   **Block Sparse Row (BSR)** compression to reduce memory bandwidth consumption.
*   **Custom DMA Controller** with double-buffered on-chip memory (SystemVerilog).
*   **Performance Counters** implementing a Roofline Model to analyze compute vs. memory bounds.

I believe my practical experience with SystemVerilog, AXI4 protocols, and performance analysis aligns well with your group's work in robust hardware design.

I have attached my CV and a brief architecture document of the accelerator. I would welcome the chance to discuss how I could contribute to your ongoing projects.

Best regards,

**Josh Carter**
[Link to GitHub/Portfolio]
[Link to LinkedIn]

---

## 🚀 LinkedIn Post Drafting

**Headline:**
Junior FPGA Engineer | Memory Architecture Enthusiast | Built Sparse INT8 Accelerator (SystemVerilog/C++)

**Project Post:**
"🚀 Just deployed my custom Sparse Matrix Accelerator on the Pynq-Z2 FPGA!

To tackle the memory wall in edge AI, I moved beyond dense matrix multiplication. My design implements **Block Sparse Row (BSR)** hardware support, allowing the accelerator to skip zero-blocks entirely and save 40%+ memory bandwidth.

**Architecture Highlights:**
🛠️ **RTL:** 14x14 Systolic Array (SystemVerilog)
⚡ **Memory:** Custom AXI4 DMA & Double-Buffered On-Chip SRAM
📊 **Analysis:** Embedded detailed performance counters to derive Roofline Models
🖥️ **Driver:** C++ Runtime for Pynq (ARM Cortex-A9)

It’s working fully on hardware—not just simulation! Check out the architecture docs and code here: [GitHub Link]

#FPGA #SystemVerilog #ComputerArchitecture #EdgeAI #MemoryHierarchy #HardwareEngineering"
