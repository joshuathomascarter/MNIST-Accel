// UART Controller - AXI-Lite Slave
// TX/RX with 16-deep FIFOs, baud rate configuration

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
module uart_ctrl #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned CLK_FREQ = 50_000_000,
  parameter int unsigned DEFAULT_BAUD = 115_200
) (
  input  logic                    clk,
  input  logic                    rst_n,
  
  // UART I/O
  input  logic                    rx,
  output logic                    tx,
  
  // AXI-Lite Slave Interface
  input  logic                    awvalid,
  output logic                    awready,
  input  logic [ADDR_WIDTH-1:0]   awaddr,
  input  logic [2:0]              awsize,
  input  logic [1:0]              awburst,
  input  logic [3:0]              awid,
  
  input  logic                    wvalid,
  output logic                    wready,
  input  logic [DATA_WIDTH-1:0]   wdata,
  input  logic [DATA_WIDTH/8-1:0] wstrb,
  input  logic                    wlast,
  
  output logic                    bvalid,
  input  logic                    bready,
  output logic [1:0]              bresp,
  output logic [3:0]              bid,
  
  input  logic                    arvalid,
  output logic                    arready,
  input  logic [ADDR_WIDTH-1:0]   araddr,
  input  logic [2:0]              arsize,
  input  logic [1:0]              arburst,
  input  logic [3:0]              arid,
  
  output logic                    rvalid,
  input  logic                    rready,
  output logic [DATA_WIDTH-1:0]   rdata,
  output logic [1:0]              rresp,
  output logic [3:0]              rid,
  output logic                    rlast,
  
  output logic                    irq_o
);

  // Register offsets
  localparam logic [7:0] TX_DATA  = 8'h00;
  localparam logic [7:0] RX_DATA  = 8'h04;
  localparam logic [7:0] STATUS   = 8'h08;
  localparam logic [7:0] CTRL     = 8'h0C;

  // Status bit definitions
  localparam int unsigned TXFIFO_FULL  = 0;
  localparam int unsigned RXFIFO_EMPTY = 1;
  localparam int unsigned TX_BUSY      = 2;

  // Baud rate calculation
  logic [15:0] baud_divisor;
  localparam logic [15:0] DEFAULT_DIVISOR = 16'(CLK_FREQ / DEFAULT_BAUD);

  // TX FIFO (16 deep, 8-bit)
  logic [7:0] tx_fifo [0:15];
  logic [3:0] tx_wr_ptr, tx_rd_ptr;
  logic [4:0] tx_count;
  logic tx_fifo_full, tx_fifo_empty;

  // RX FIFO (16 deep, 8-bit)
  logic [7:0] rx_fifo [0:15];
  logic [3:0] rx_wr_ptr, rx_rd_ptr;
  logic [4:0] rx_count /* verilator public_flat_rw */;
  logic rx_fifo_full, rx_fifo_empty;

  // TX shift register
  logic [9:0] tx_shift;  // [0]=start, [1:8]=data, [9]=stop
  logic [3:0] tx_bit_cnt;
  logic tx_busy;
  logic [15:0] tx_baud_cnt;
  logic tx_start;

  // RX shift register
  logic [9:0] rx_shift;
  logic [3:0] rx_bit_cnt;
  logic [15:0] rx_baud_cnt;
  logic [7:0] rx_data_captured /* verilator public_flat_rw */;
  logic rx_valid               /* verilator public_flat_rw */;
  logic rx_in_sync;
  logic rx_in_sync_r;

  // AXI transaction tracking
  logic [3:0] aw_id, ar_id;
  logic [7:0] ar_addr_r;
  logic b_pending, ar_valid;

  // ===== FIFO Logic =====
  
  assign tx_fifo_full  = (tx_count == 5'd16);
  assign tx_fifo_empty = (tx_count == 5'd0);
  assign rx_fifo_full  = (rx_count == 5'd16);
  assign rx_fifo_empty = (rx_count == 5'd0);
  
  // ===== TX PATH =====
  
  // TX consume signal: TX shift register picks up next byte from FIFO
  logic tx_consume;
  assign tx_consume = !tx_busy && !tx_fifo_empty;

  // TX produce signal: AXI write puts a byte into TX FIFO
  logic tx_produce /* verilator public_flat_rw */;
  assign tx_produce = awvalid && wvalid && !b_pending && awaddr[7:0] == TX_DATA && !tx_fifo_full;

  // Baud counter and bit counter for TX
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx_busy <= 1'b0;
      tx_bit_cnt <= '0;
      tx_baud_cnt <= '0;
      tx_shift <= '0;
      tx <= 1'b1;
      tx_start <= 1'b0;
      tx_rd_ptr <= '0;
    end else begin
      // Baud counter
      if (tx_start || (tx_busy && tx_baud_cnt == 0)) begin
        tx_baud_cnt <= baud_divisor - 1;
      end else if (tx_baud_cnt != 0) begin
        tx_baud_cnt <= tx_baud_cnt - 1;
      end

      // Start transmission if FIFO has data and TX is idle
      if (tx_consume) begin
        tx_busy <= 1'b1;
        tx_start <= 1'b1;
        tx_shift <= {1'b1, tx_fifo[tx_rd_ptr], 1'b0};  // [9]=stop, [8:1]=data, [0]=start
        tx_bit_cnt <= 4'd10;  // 10 bits (start + 8 data + stop)
        tx_rd_ptr <= tx_rd_ptr + 1;
      end else begin
        tx_start <= 1'b0;
      end

      // Shift out bits
      if (tx_busy && tx_baud_cnt == 0) begin
        tx <= tx_shift[0];
        tx_shift <= {1'b1, tx_shift[9:1]};
        tx_bit_cnt <= tx_bit_cnt - 1;
        if (tx_bit_cnt == 1) begin
          tx_busy <= 1'b0;
        end
      end
    end
  end

  // ===== RX PATH =====
  
  // Sync RX input
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_in_sync <= 1'b1;
      rx_in_sync_r <= 1'b1;
    end else begin
      rx_in_sync_r <= rx;
      rx_in_sync <= rx_in_sync_r;
    end
  end

  // RX deserializer. Sample the first data bit 1.5 bit-times after the start
  // edge, then sample each following bit one full bit period later.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_bit_cnt <= '0;
      rx_baud_cnt <= '0;
      rx_shift <= '0;
      rx_data_captured <= '0;
      rx_valid <= 1'b0;
    end else begin
      rx_valid <= 1'b0;

      // Detect start bit (falling edge on rx)
      if (rx_bit_cnt == 0 && rx_in_sync_r == 1'b0 && rx_in_sync == 1'b1) begin
        rx_bit_cnt <= 4'd8;
        rx_baud_cnt <= baud_divisor + (baud_divisor >> 1) - 1;
        rx_shift <= '0;
      end else if (rx_bit_cnt != 0) begin
        if (rx_baud_cnt == 0) begin
          rx_baud_cnt <= baud_divisor - 1;
          if (rx_bit_cnt == 1) begin
            rx_data_captured <= {rx_in_sync, rx_shift[9:3]};
            rx_valid <= 1'b1;
            rx_bit_cnt <= '0;
          end else begin
            rx_shift <= {rx_in_sync, rx_shift[9:1]};
            rx_bit_cnt <= rx_bit_cnt - 1;
          end
        end else begin
          rx_baud_cnt <= rx_baud_cnt - 1;
        end
      end
    end
  end

  // RX FIFO write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_wr_ptr <= '0;
    end else begin
      if (rx_valid && !rx_fifo_full) begin
        rx_fifo[rx_wr_ptr] <= rx_data_captured;
        rx_wr_ptr <= rx_wr_ptr + 1;
      end
    end
  end

  // Single always_ff owns rx_count to avoid multi-driver
  // rx_count decrements at the R channel (rvalid+rready) to match rx_rd_ptr advancement
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_count <= '0;
    end else begin
      unique case ({rx_valid && !rx_fifo_full,
                    rvalid && rready && ar_addr_r == RX_DATA && !rx_fifo_empty})
        2'b10:   rx_count <= rx_count + 1;  // received only
        2'b01:   rx_count <= rx_count - 1;  // read only
        2'b11:   rx_count <= rx_count;       // simultaneous receive + read = net zero
        default: rx_count <= rx_count;
      endcase
    end
  end

  // ===== AXI WRITE PATH =====
  
  assign awready = !b_pending;
  assign wready  = awvalid && !b_pending;

  // Write transaction handler — single always_ff owns baud_divisor
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      baud_divisor <= DEFAULT_DIVISOR;
      b_pending <= 1'b0;
      aw_id    <= '0;
      tx_wr_ptr <= '0;
    end else begin
      if (b_pending && bready) begin
        b_pending <= 1'b0;
      end else if (awvalid && wvalid && !b_pending) begin
        b_pending <= 1'b1;
        aw_id <= awid;
        case (awaddr[7:0])
          TX_DATA: begin
            if (!tx_fifo_full) begin
              tx_fifo[tx_wr_ptr] <= wdata[7:0];
              tx_wr_ptr <= tx_wr_ptr + 1;
            end
          end
          CTRL: begin
            // CTRL[0] = TX enable, CTRL[1] = RX enable (baud_divisor unchanged)
            // To change baud, write new divisor to CTRL[31:16]
            if (wdata[31:16] != 16'd0)
              baud_divisor <= wdata[31:16];
          end
          default: begin
            // Read-only or undefined
          end
        endcase
      end
    end
  end

  assign bvalid = b_pending;
  assign bresp = 2'b00;
  assign bid = aw_id;

  // TX FIFO count: handles both produce (AXI write) and consume (TX shift register)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx_count <= '0;
    end else begin
      unique case ({tx_produce, tx_consume})
        2'b10:   tx_count <= tx_count + 1;
        2'b01:   tx_count <= tx_count - 1;
        default: tx_count <= tx_count;
      endcase
    end
  end

  // ===== AXI READ PATH =====
  
  assign arready = 1'b1;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ar_valid <= 1'b0;
      ar_id    <= '0;
      ar_addr_r <= '0;
      rdata    <= '0;
    end else begin
      if (arvalid && arready) begin
        ar_valid  <= 1'b1;
        ar_id     <= arid;
        ar_addr_r <= araddr[7:0];
        // Capture rdata at AR handshake using OLD rx_rd_ptr (before R-channel advance)
        case (araddr[7:0])
          TX_DATA: rdata <= {24'b0, tx_fifo[tx_rd_ptr]};
          RX_DATA: rdata <= {24'b0, rx_fifo[rx_rd_ptr]};
          STATUS:  rdata <= {29'b0, tx_busy, rx_fifo_empty, tx_fifo_full};
          CTRL:    rdata <= {16'b0, baud_divisor};
          default: rdata <= '0;
        endcase
      end else if (rvalid && rready) begin
        ar_valid <= 1'b0;
      end
    end
  end

  // Read response
  assign rvalid = ar_valid;
  assign rid = ar_id;
  assign rresp = 2'b00;
  assign rlast = 1'b1;

  // RX FIFO read pointer advances at R channel (after rdata has been captured)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_rd_ptr <= '0;
    end else begin
      if (rvalid && rready && ar_addr_r == RX_DATA && !rx_fifo_empty) begin
        rx_rd_ptr <= rx_rd_ptr + 1;
      end
    end
  end

  // Interrupt when RX FIFO has data
  assign irq_o = !rx_fifo_empty;

endmodule : uart_ctrl
