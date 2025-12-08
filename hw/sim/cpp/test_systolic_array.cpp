#include <iostream>
#include <cstdint>
#include <vector>
// #include <verilated.h> // Verilator Library (commented out for now)
// #include <verilated_cov.h> // Verilator Coverage (commented out for now)
// #include "Vpe.h"       // The Verilog Model (Generated later)

// ==========================================
// 1. The Custom Stack Allocator (Unchanged)
// ==========================================
class StackAllocator {
public:
    StackAllocator(size_t size_bytes) {
        total_size = size_bytes;
        memory_pool = new uint8_t[size_bytes];
        offset = 0;
    }

    ~StackAllocator() {
        delete[] memory_pool;
    }

    template <typename T>
    T* alloc(size_t count = 1) {
        size_t size_needed = sizeof(T) * count;
        if (offset + size_needed > total_size) return nullptr;
        void* ptr = memory_pool + offset;
        offset += size_needed;
        return static_cast<T*>(ptr);
    }

    void reset() {
        offset = 0;
    }

private:
    uint8_t* memory_pool;
    size_t total_size;
    size_t offset;
};

// ==========================================
// 2. The Tensor Class (Matrix2D)
// ==========================================
template <typename T>
class Matrix2D {
public:
    int rows;
    int cols;
    T* data; // The raw pointer to the flat memory

    // Constructor: Asks the allocator for memory
    Matrix2D(StackAllocator& allocator, int r, int c) {
        rows = r;
        cols = c;
        // Ask for enough space for the whole matrix (r * c)
        data = allocator.alloc<T>(rows * cols);
    }

    // The "Accessor" - Maps (row, col) to 1D index
    // Returns a reference (&) so we can read AND write: mat.at(0,0) = 5;
    T& at(int r, int c) {
        // ROW-MAJOR MAPPING:
        // Skip 'r' full rows (r * cols), then add 'c' steps.
        return data[r * cols + c];
    }

    // Helper to fill with random data
    void fill_random() {
        for (int i = 0; i < rows * cols; i++) {
            data[i] = static_cast<T>(rand() % 10); // Random 0-9
        }
    }
    
    // Helper to print for debugging
    void print(const char* name) {
        std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):\n";
        for(int r=0; r<rows; r++) {
            for(int c=0; c<cols; c++) {
                std::cout << (int)at(r,c) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

// ==========================================
// 3. The Skew Layout Function
// ==========================================
// This function rearranges the data to match the "Wavefront" timing.
// If Row 0 starts at T=0, Row 1 starts at T=1, Row 2 at T=2...
// We need to pad the input stream with zeros (bubbles) so they align correctly.
std::vector<uint8_t> pack_skewed_input(Matrix2D<uint8_t>& mat) {
    int rows = mat.rows;
    int cols = mat.cols;
    
    // The total time needed is cols (width of matrix) + rows (skew delay)
    int total_cycles = cols + rows; 
    
    // We will return a flat vector representing the input bus at each cycle.
    // Assuming the bus width is 'rows' bytes (one byte per row per cycle).
    std::vector<uint8_t> skewed_stream(total_cycles * rows, 0); // Init with 0

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // The "Time" this specific pixel enters the array:
            // It enters at column index 'c', PLUS the skew delay for its row 'r'.
            int cycle_time = c + r; 
            
            // Where in the flat stream does this byte go?
            // Index = (Time * BusWidth) + RowIndex
            int flat_index = (cycle_time * rows) + r;
            
            skewed_stream[flat_index] = mat.at(r, c);
        }
    }
    return skewed_stream;
}

int main() {
    StackAllocator mem(1024 * 1024); // 1MB Heap
    
    // 1. Create a 4x4 Matrix on the Stack Allocator
    Matrix2D<uint8_t> input_matrix(mem, 4, 4);
    input_matrix.fill_random();
    input_matrix.print("Input");

    // 2. Pack it into the Skewed Layout
    std::vector<uint8_t> stream = pack_skewed_input(input_matrix);

    // 3. Verify the Skew
    std::cout << "Skewed Stream (Cycle by Cycle):\n";
    for(int t=0; t<8; t++) { // 4 cols + 4 rows skew = 8 cycles
        std::cout << "Cycle " << t << ": ";
        for(int r=0; r<4; r++) {
            // Print the byte for each row at this time step
            int val = stream[t*4 + r];
            // A bubble exists if the row hasn't started yet (t < r means Cycle is before Row's start time)
            if(t < r) std::cout << ". ";  // This row hasn't started yet - definite bubble
            else std::cout << val << " ";  // Row has started - print actual value (even if 0)
        }
        std::cout << "\n";
    }

    return 0;
}