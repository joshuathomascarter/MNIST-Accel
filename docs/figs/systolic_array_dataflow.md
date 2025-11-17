# Systolic Array Dataflow Diagram

```
ACCEL-v1 Row-Stationary Systolic Array (2x2 Example)
=====================================================

Time t=0: Initial State
┌─────────────────────────────────────────────────────────────────┐
│                        Weight Loading Phase                     │
└─────────────────────────────────────────────────────────────────┘

    W[0,0]   W[0,1]     ← Weights loaded from north (column-wise)
      ↓        ↓
   ┌─────┐  ┌─────┐
   │PE00 │  │PE01 │     Row 0
   └─────┘  └─────┘
      ↓        ↓
   ┌─────┐  ┌─────┐
   │PE10 │  │PE11 │     Row 1  
   └─────┘  └─────┘
      ↓        ↓
    W[1,0]   W[1,1]     ← Weights flow down (stationary in PEs)

═══════════════════════════════════════════════════════════════════

Time t=1: Computation Begins (k=0)
┌─────────────────────────────────────────────────────────────────┐
│                    Activation Flow Phase                        │
└─────────────────────────────────────────────────────────────────┘

            W₀₀      W₀₁         ← Weights STATIONARY in PEs
              ↓        ↓
A[0,0] ────▶┌─────┐  ┌─────┐ ────▶ A[0,0] flows right
            │ PE  │  │ PE  │
            │ 00  │  │ 01  │      MAC: ACC₀₀ += A₀₀×W₀₀
            └─────┘  └─────┘             ACC₀₁ += A₀₀×W₀₁
              ↓        ↓
A[1,0] ────▶┌─────┐  ┌─────┐ ────▶ A[1,0] flows right  
            │ PE  │  │ PE  │
            │ 10  │  │ 11  │      MAC: ACC₁₀ += A₁₀×W₀₀
            └─────┘  └─────┘             ACC₁₁ += A₁₀×W₀₁
              ↓        ↓
            W₁₀      W₁₁         ← Weights flow down to next row

═══════════════════════════════════════════════════════════════════

Time t=2: Next Activation Set (k=1)
┌─────────────────────────────────────────────────────────────────┐
│                 Accumulation Continues                          │
└─────────────────────────────────────────────────────────────────┘

            W₀₀      W₀₁         ← SAME weights (reused!)
              ↓        ↓
A[0,1] ────▶┌─────┐  ┌─────┐ ────▶ Different activations
            │ PE  │  │ PE  │
            │ 00  │  │ 01  │      MAC: ACC₀₀ += A₀₁×W₀₀
            └─────┘  └─────┘             ACC₀₁ += A₀₁×W₀₁
              ↓        ↓
A[1,1] ────▶┌─────┐  ┌─────┐ ────▶ (Accumulate with previous)
            │ PE  │  │ PE  │
            │ 10  │  │ 11  │      MAC: ACC₁₀ += A₁₁×W₀₀  
            └─────┘  └─────┘             ACC₁₁ += A₁₁×W₀₁
              ↓        ↓
            W₁₀      W₁₁         ← Same weights propagated

═══════════════════════════════════════════════════════════════════

Final Result: C = A × B
┌─────────────────────────────────────────────────────────────────┐
│              Accumulated Partial Sums                           │
└─────────────────────────────────────────────────────────────────┘

C[0,0] = ACC₀₀ = Σ(k) A[0,k] × W[k,0]    C[0,1] = ACC₀₁ = Σ(k) A[0,k] × W[k,1]
C[1,0] = ACC₁₀ = Σ(k) A[1,k] × W[k,0]    C[1,1] = ACC₁₁ = Σ(k) A[1,k] × W[k,1]

Key Benefits of Row-Stationary Dataflow:
▪ Weights loaded ONCE, reused for multiple activations
▪ Activations flow through array (minimal storage)
▪ High compute-to-communication ratio
▪ Efficient for CNN workloads (filter reuse)
```

## Data Type Flow Diagram

```
INT8 → INT32 → INT8 Quantization Pipeline
=========================================

Input Matrices (Host)          Hardware Computation         Output (Host)
┌─────────────────┐            ┌─────────────────┐          ┌──────────────┐
│   A[M×K] INT8   │ ───────▶   │                 │          │              │
│   B[K×N] INT8   │ UART       │  Systolic Array │ UART     │  C[M×N] INT8 │
│   Scales: Sa,Sw │ Protocol   │                 │ Protocol │  Scale: Sc   │
└─────────────────┘            │ ┌─────────────┐ │          └──────────────┘
                               │ │PE  PE  PE..│ │                │
┌─────────────────┐            │ │PE  PE  PE..│ │          ┌──────────────┐
│ Pre-Processing  │            │ │PE  PE  PE..│ │          │Post-Process  │
│ • Load Image    │            │ └─────────────┘ │          │• Dequantize  │
│ • Normalize     │            │                 │          │• Apply Scale │
│ • Quantize      │            │ INT32 Accumulat.│          │• Clamp Range │
│ • Apply Scale   │            │ C_int32[M×N]    │          │• Return FP32 │
└─────────────────┘            └─────────────────┘          └──────────────┘
         │                              │                          ▲
         ▼                              ▼                          │
┌─────────────────┐            ┌─────────────────┐          ┌──────────────┐
│ A_quantized     │            │ MAC Operations  │          │ C_quantized  │
│ = round(A/Sa)   │            │ PE[i,j].acc +=  │          │ = C_int32 *  │
│ ∈ [-128, +127]  │            │   A_int8[i,k] × │          │   (Sa×Sw/Sc) │
└─────────────────┘            │   B_int8[k,j]   │          └──────────────┘
                               │ ∈ [-2²³, +2²³]  │
┌─────────────────┐            └─────────────────┘
│ B_quantized     │                     │
│ = round(B/Sw)   │                     ▼
│ ∈ [-128, +127]  │            ┌─────────────────┐
└─────────────────┘            │ Requantization  │
                               │ • Multiply Scale│
                               │ • Shift Right   │
                               │ • Saturate      │
                               └─────────────────┘

Precision Analysis:
▪ Input:  8-bit integers (INT8)
▪ Compute: 32-bit accumulation (INT32) prevents overflow
▪ Output: 8-bit integers (INT8) with proper scaling
▪ Accuracy: ~1% degradation vs FP32 for most CNNs
```