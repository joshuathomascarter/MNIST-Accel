# Email to Prof. Sofiène Tahar

**To:** tahar@ece.concordia.ca  
**Subject:** Open-Source Sparse ML Accelerator — Interest in RA Position (Sept 2026)

---

Dear Professor Tahar,

I am writing to share an open-source hardware project I have been building and to express my interest in a Research Assistant position in your group starting September 2026.

Over the past several months I have designed and implemented a complete sparse-matrix ML inference accelerator targeting the Xilinx Zynq-7020 (PYNQ-Z2). The system includes:

- A 14×14 weight-stationary systolic array with INT8 MAC units
- Block Sparse Row (BSR) storage with zero-skip logic for structured sparsity
- An AXI4 DMA bridge for PS–PL data movement
- A full software stack: PyTorch quantization and BSR export, C++ host driver, and Python golden-model verification
- Yosys + Verilator synthesis and simulation, with cocotb-based testbenches

The entire project — RTL, software, tests, and documentation — is available on GitHub. Everything builds and passes from a single clone: `pytest` for the ML pipeline, `ctest` for the C++ driver, and `make` for the RTL simulation.

Your work on the formal verification of SystemC transaction-level models (IEEE TVLSI, 2006) caught my attention because my accelerator's AXI interface operates at a similar abstraction boundary — transactions that must maintain ordering, burst-length, and address-alignment invariants that are difficult to verify by simulation alone. Likewise, your comparative study of HOL and MDG on the Fairisle ATM switch fabric (Nordic Journal of Computing, 1999) addresses exactly the class of routing and arbitration properties I expect to encounter in my next project.

That next step is a sparsity-aware Network-on-Chip for tiled ML accelerators. The idea is straightforward: a NoC scheduler that reads BSR metadata at tile-dispatch time and dynamically allocates virtual channels and bandwidth — dense tiles get full bandwidth while sparse tiles share fewer channels. The router uses deterministic XY wormhole routing on a 2D mesh. I have not seen anyone combine sparsity metadata with NoC scheduling in this way, and the architecture raises properties that need formal treatment: deadlock freedom under mixed traffic, livelock avoidance when sparse tiles complete out of order, and virtual-channel allocation fairness. These are the kinds of properties your group has verified with HOL and MDG on network fabrics, and I would value the opportunity to work under your supervision to bring that rigor to this design.

I will have working RTL for the NoC router and a draft paper ready to share before any application deadline. I would be glad to discuss any of this further at your convenience.

Thank you for your time.

Best regards,  
Joshua Carter

GitHub: https://github.com/joshcarter  
