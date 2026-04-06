# AXI Crossbar Line-by-Line Walkthrough

This file explains [hw/rtl/top/axi_crossbar.sv](hw/rtl/top/axi_crossbar.sv) in plain language, but still keeps the actual HDL structure visible.

The goal of this module is simple:

- take requests from 2 masters
- decide which of 8 slaves each master wants
- arbitrate if multiple masters want the same slave
- forward the winning master's signals to that slave
- route ready and response signals back to the right master

## Big Picture First

Think of the crossbar as five stages:

1. Decode each master's address into a target slave.
2. For each slave, decide which master wins if there is contention.
3. Forward the winning master's address and data to the slave.
4. Send ready signals back to the correct master.
5. Send read and write responses back to the correct master.

In other words:

```text
master address -> decode target slave -> arbitrate at that slave -> mux winner forward -> route response back
```

## AXI Naming Cheat Sheet

- `AW` = write address channel
- `W` = write data channel
- `B` = write response channel
- `AR` = read address channel
- `R` = read data channel

Prefix meaning:

- `m_` means master-side signal
- `s_` means slave-side signal

Examples:

- `m_awaddr[1]` means write address coming from master 1
- `s_awaddr[4]` means write address going out to slave 4

## Important Syntax Before Going Line by Line

### Packed arrays like this

```systemverilog
logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] m_awaddr;
```

Read this as:

- there is one `ADDR_WIDTH`-wide address per master
- mentally use it as `m_awaddr[master_index]`

### Unpacked arrays like this

```systemverilog
logic [NUM_SLAVES-1:0] m_aw_target [NUM_MASTERS];
```

Read this as:

- for each master, there is a one-hot slave target vector
- mentally use it as `m_aw_target[master][slave]`

### Generate loops

```systemverilog
generate
  for (...) begin : some_block
```

This does not loop at runtime like software.
It creates repeated hardware instances when the design is elaborated.

## Line-by-Line Walkthrough

## Header Comment Block

```systemverilog
// ===========================================================================
// axi_crossbar.sv — AXI Crossbar: 2 Masters × 8 Slaves
// ===========================================================================
// Instantiates axi_addr_decoder (per-master) and axi_arbiter (per-slave)
// as standalone sub-modules, then wires the datapath MUX.
//
// Refactored from monolithic inline logic for testability and reuse.
// ===========================================================================
```

This tells you what the file is trying to be:

- a crossbar between 2 masters and 8 slaves
- address decoding is split out into a reusable module
- arbitration is split out into a reusable module
- this file mostly wires those pieces together

So this file is not implementing everything from scratch. It is orchestrating sub-blocks.

## Module Declaration and Parameters

```systemverilog
module axi_crossbar #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned ID_WIDTH = 4,
  parameter int unsigned NUM_MASTERS = 2,
  parameter int unsigned NUM_SLAVES = 8
) (
```

This is the module header.

Each parameter controls the structure of the generated hardware.

- `ADDR_WIDTH = 32`
  The address bus is 32 bits wide.
- `DATA_WIDTH = 32`
  The data bus is 32 bits wide.
- `ID_WIDTH = 4`
  AXI IDs are 4 bits wide.
- `NUM_MASTERS = 2`
  There are 2 request sources.
- `NUM_SLAVES = 8`
  There are 8 possible destination slaves.

This means the module is parameterized, but this particular design is intended for 2-by-8.

## Clock and Reset

```systemverilog
  input  logic                     clk,
  input  logic                     rst_n,
```

These are the normal sequential control inputs.

- `clk` is the clock.
- `rst_n` is active-low reset.

This crossbar itself contains mostly combinational routing, but the arbiters instantiated later use `clk` and `rst_n` because round-robin state must be remembered.

## Master-Side Write Address Channel

```systemverilog
  input  logic [NUM_MASTERS-1:0]   m_awvalid,
  output logic [NUM_MASTERS-1:0]   m_awready,
  input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] m_awaddr,
  input  logic [NUM_MASTERS-1:0][ID_WIDTH-1:0]   m_awid,
```

These are the write-address signals coming from the masters.

- `m_awvalid[master]`
  That master has a valid write address request.
- `m_awready[master]`
  The crossbar is ready to accept that master's write address.
- `m_awaddr[master]`
  The actual write address from that master.
- `m_awid[master]`
  The AXI transaction ID from that master.

Example:

```text
m_awvalid = 2'b10
m_awaddr[1] = 32'h4000_0100
```

This means master 1 is presenting a valid write address to the crossbar.

## Master-Side Write Data Channel

```systemverilog
  input  logic [NUM_MASTERS-1:0]   m_wvalid,
  output logic [NUM_MASTERS-1:0]   m_wready,
  input  logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] m_wdata,
  input  logic [NUM_MASTERS-1:0][DATA_WIDTH/8-1:0] m_wstrb,
  input  logic [NUM_MASTERS-1:0]   m_wlast,
```

These are the write-data signals from the masters.

- `m_wvalid[master]`
  That master has write data ready.
- `m_wready[master]`
  The crossbar is ready to accept write data from that master.
- `m_wdata[master]`
  The actual write payload.
- `m_wstrb[master]`
  Byte-enable strobes for the write.
- `m_wlast[master]`
  Marks the last beat of a burst.

This file routes write data by following the AW grant. That is an architectural simplification worth remembering.

## Master-Side Write Response Channel

```systemverilog
  output logic [NUM_MASTERS-1:0]   m_bvalid,
  input  logic [NUM_MASTERS-1:0]   m_bready,
  output logic [NUM_MASTERS-1:0][1:0] m_bresp,
  output logic [NUM_MASTERS-1:0][ID_WIDTH-1:0] m_bid,
```

These are the write-response signals back to the masters.

- `m_bvalid[master]`
  The crossbar has a valid write response for that master.
- `m_bready[master]`
  That master is ready to accept the write response.
- `m_bresp[master]`
  Response status.
- `m_bid[master]`
  Response ID.

These come from the selected slave and are routed back in section 5.

## Master-Side Read Address Channel

```systemverilog
  input  logic [NUM_MASTERS-1:0]   m_arvalid,
  output logic [NUM_MASTERS-1:0]   m_arready,
  input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] m_araddr,
  input  logic [NUM_MASTERS-1:0][ID_WIDTH-1:0]   m_arid,
```

These are the read-address signals from the masters.

Exactly like AW, but for reads.

- `m_arvalid[master]`
- `m_arready[master]`
- `m_araddr[master]`
- `m_arid[master]`

## Master-Side Read Data Channel

```systemverilog
  output logic [NUM_MASTERS-1:0]   m_rvalid,
  input  logic [NUM_MASTERS-1:0]   m_rready,
  output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] m_rdata,
  output logic [NUM_MASTERS-1:0][1:0] m_rresp,
  output logic [NUM_MASTERS-1:0][ID_WIDTH-1:0] m_rid,
  output logic [NUM_MASTERS-1:0]   m_rlast,
```

These are the read-data signals back to the masters.

- `m_rvalid[master]`
- `m_rready[master]`
- `m_rdata[master]`
- `m_rresp[master]`
- `m_rid[master]`
- `m_rlast[master]`

Again, these are routed back from the selected slave later in section 5.

## Slave-Side Port Lists

```systemverilog
  output logic [NUM_SLAVES-1:0]    s_awvalid,
  input  logic [NUM_SLAVES-1:0]    s_awready,
  output logic [NUM_SLAVES-1:0][ADDR_WIDTH-1:0] s_awaddr,
  output logic [NUM_SLAVES-1:0][ID_WIDTH-1:0]   s_awid,
```

This is the write-address interface going out to each slave.

There is one set of AW signals per slave.

Likewise the next blocks define:

- slave-side write data
- slave-side write response
- slave-side read address
- slave-side read data

The key mental model is this:

- `m_*` comes into the crossbar from masters
- `s_*` leaves the crossbar toward slaves

Then the response direction is reversed:

- `s_bvalid`, `s_rvalid`, `s_rdata`, etc. come into the crossbar from slaves
- `m_bvalid`, `m_rvalid`, `m_rdata`, etc. leave the crossbar toward masters

## Section 1: Address Decode

```systemverilog
  logic [NUM_SLAVES-1:0] m_aw_target [NUM_MASTERS];
  logic [NUM_SLAVES-1:0] m_ar_target [NUM_MASTERS];
```

These are internal destination vectors.

Interpret them as:

- `m_aw_target[master][slave]`
- `m_ar_target[master][slave]`

For each master, these store a one-hot answer to the question:

"Which slave does this address belong to?"

Example:

```text
m_awaddr[1] = 32'h4000_0100
```

The decoder sees top nibble `4`, so the result becomes conceptually:

```text
m_aw_target[1] = one-hot vector selecting slave 4
```

```systemverilog
  genvar mi;
  generate
    for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_decoders
```

This generates one decode block per master.

`mi` is not a runtime variable. It is only used while building the hardware instances.

```systemverilog
      axi_addr_decoder #(
        .ADDR_WIDTH(ADDR_WIDTH), .NUM_SLAVES(NUM_SLAVES)
      ) u_aw_dec (
        .addr        (m_awaddr[mi]),
        .slave_sel   (m_aw_target[mi]),
        .decode_error(/* unused — error sink is slave 7 */)
      );
```

For each master:

- take its AW address
- decode the address into a one-hot slave selection
- store the result in `m_aw_target[mi]`

`decode_error` is ignored here because this design routes unknown regions to the final error sink slave.

```systemverilog
      axi_addr_decoder #(
        .ADDR_WIDTH(ADDR_WIDTH), .NUM_SLAVES(NUM_SLAVES)
      ) u_ar_dec (
        .addr        (m_araddr[mi]),
        .slave_sel   (m_ar_target[mi]),
        .decode_error()
      );
```

Same thing for reads.

So after section 1, the crossbar knows for every master:

- which slave its write address wants
- which slave its read address wants

## Section 2: Per-Slave Arbitration

```systemverilog
  logic [NUM_MASTERS-1:0] aw_grant [NUM_SLAVES];
  logic [NUM_MASTERS-1:0] ar_grant [NUM_SLAVES];
```

These store who won for each slave.

Interpret them as:

- `aw_grant[slave][master]`
- `ar_grant[slave][master]`

Each one is one-hot across masters.

Example:

```text
aw_grant[4] = 2'b10
```

means master 1 won the write-address arbitration for slave 4.

```systemverilog
  localparam int unsigned MIDX_W = $clog2(NUM_MASTERS);
  logic [MIDX_W-1:0]     aw_grant_idx [NUM_SLAVES];
  logic [MIDX_W-1:0]     ar_grant_idx [NUM_SLAVES];
```

These hold the same winner information, but as a binary master number instead of one-hot.

Example with 2 masters:

- `aw_grant[4] = 2'b10`
- `aw_grant_idx[4] = 1`

The index form is convenient for direct array indexing like `m_awaddr[aw_grant_idx[si]]`.

```systemverilog
  genvar si;
  generate
    for (si = 0; si < NUM_SLAVES; si++) begin : gen_arbiters
```

This generates one arbitration block per slave.

That means every slave independently decides which master may talk to it.

```systemverilog
      logic [NUM_MASTERS-1:0] aw_req, ar_req;
```

These are local request vectors for this particular slave.

For slave `si`:

- `aw_req[master]` means that master wants this slave on AW
- `ar_req[master]` means that master wants this slave on AR

```systemverilog
      for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_req
        assign aw_req[mi] = m_aw_target[mi][si] && m_awvalid[mi];
        assign ar_req[mi] = m_ar_target[mi][si] && m_arvalid[mi];
      end
```

This is one of the most important lines in the file.

It says a master is requesting slave `si` only if:

- the decoder said that slave `si` is its target
- and the master is actually asserting valid on that channel

So `aw_req[mi]` is not just “master `mi` has an address.”
It means “master `mi` has an address for this exact slave.”

Example:

- master 0 wants slave 4 and is valid
- master 1 wants slave 4 and is valid

Then for `si = 4`:

```text
aw_req = 2'b11
```

which means both masters are contending for slave 4.

For some other slave like `si = 2`, the request vector might be:

```text
aw_req = 2'b00
```

meaning nobody wants slave 2 right now.

```systemverilog
      axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_aw_arb (
        .clk            (clk),
        .rst_n          (rst_n),
        .req            (aw_req),
        .handshake_done (s_awvalid[si] && s_awready[si]),
        .grant          (aw_grant[si]),
        .grant_idx      (aw_grant_idx[si])
      );
```

This instantiates the write-address arbiter for slave `si`.

Meaning:

- look at which masters want this slave on AW
- choose one using round-robin fairness
- remember who should get priority next time using `clk` and `rst_n`

`handshake_done` for AW is:

```text
s_awvalid[si] && s_awready[si]
```

That means the arbiter rotates only after a real AW transfer to that slave completes.

```systemverilog
      axi_arbiter #(.NUM_MASTERS(NUM_MASTERS)) u_ar_arb (
        .clk            (clk),
        .rst_n          (rst_n),
        .req            (ar_req),
        .handshake_done (s_arvalid[si] && s_arready[si]),
        .grant          (ar_grant[si]),
        .grant_idx      (ar_grant_idx[si])
      );
```

Same logic for reads.

At the end of section 2, each slave knows:

- which master may issue AW to it
- which master may issue AR to it

## Section 3: Datapath Mux

This section takes the arbitration results and forwards the winning master's signals to the slave.

```systemverilog
  generate
    for (si = 0; si < NUM_SLAVES; si++) begin : gen_slave_mux
```

Again, one repeated forwarding block per slave.

### Write Address Mux

```systemverilog
      always_comb begin
        s_awvalid[si] = |aw_grant[si];
        s_awaddr[si]  = m_awaddr[aw_grant_idx[si]];
        s_awid[si]    = m_awid[aw_grant_idx[si]];
      end
```

This means:

- if any master won AW for slave `si`, assert `s_awvalid[si]`
- use the winning master index to choose which address and ID to drive to the slave

Example:

```text
aw_grant[4] = 2'b10
aw_grant_idx[4] = 1
```

Then:

- `s_awvalid[4] = 1`
- `s_awaddr[4] = m_awaddr[1]`
- `s_awid[4]   = m_awid[1]`

That means slave 4 is now seeing master 1's write address channel.

### Write Data Mux

```systemverilog
      always_comb begin
        s_wvalid[si] = 1'b0;
        s_wdata[si]  = '0;
        s_wstrb[si]  = '0;
        s_wlast[si]  = 1'b0;
        if (|aw_grant[si]) begin
          s_wvalid[si] = m_wvalid[aw_grant_idx[si]];
          s_wdata[si]  = m_wdata[aw_grant_idx[si]];
          s_wstrb[si]  = m_wstrb[aw_grant_idx[si]];
          s_wlast[si]  = m_wlast[aw_grant_idx[si]];
        end
      end
```

This block does two things.

First it gives safe defaults:

- invalid by default
- zero data by default

Then if some master won AW for this slave, it forwards that master's W signals.

This design assumes write data follows the AW ownership.

That means once slave `si` decided which master owns the write path, the W channel for that slave is taken from the same winning master.

Example:

```text
aw_grant_idx[4] = 1
```

Then:

- `s_wvalid[4] = m_wvalid[1]`
- `s_wdata[4]  = m_wdata[1]`
- `s_wstrb[4]  = m_wstrb[1]`
- `s_wlast[4]  = m_wlast[1]`

### Read Address Mux

```systemverilog
      always_comb begin
        s_arvalid[si] = |ar_grant[si];
        s_araddr[si]  = m_araddr[ar_grant_idx[si]];
        s_arid[si]    = m_arid[ar_grant_idx[si]];
      end
```

Same idea as the AW mux, but for read addresses.

If slave `si` granted read access to master `k`, then slave `si` sees:

- `m_araddr[k]`
- `m_arid[k]`

### Slave-Side Ready for Responses

```systemverilog
      assign s_bready[si] = |aw_grant[si] ? m_bready[aw_grant_idx[si]] : m_bready[0];
      assign s_rready[si] = |ar_grant[si] ? m_rready[ar_grant_idx[si]] : m_rready[0];
```

These lines tell each slave whether the destination master is ready to accept a response.

Meaning:

- if slave `si` currently belongs to some winning AW master, use that master's `bready`
- if slave `si` currently belongs to some winning AR master, use that master's `rready`

The fallback `m_bready[0]` and `m_rready[0]` values just keep the signal driven when no grant exists.

The meaningful case is the granted case.

## Section 4: Ready Back-Propagation

This section sends `ready` from the slave side back to the correct master.

```systemverilog
  always_comb begin
    m_awready = '0;
    m_wready  = '0;
    m_arready = '0;
```

Start by clearing all ready outputs.

That prevents stale values or accidental multiple assignments.

```systemverilog
    for (int m = 0; m < NUM_MASTERS; m++) begin
      for (int s = 0; s < NUM_SLAVES; s++) begin
```

This loops through every master/slave combination.

The logic is asking:

- does master `m` target slave `s`?
- did slave `s` grant access to master `m`?

If both are true, then that slave's ready must be returned to that master.

```systemverilog
        if (m_aw_target[m][s] && aw_grant[s][m])
          m_awready[m] = s_awready[s];
```

This means:

- master `m` gets write-address ready from slave `s`
- only if slave `s` is actually the target and actually granted that master

So `m_awready[m]` is not a global OR of every slave.
It is specifically taken from the slave that this master won.

```systemverilog
        if (m_aw_target[m][s] && aw_grant[s][m])
          m_wready[m] = s_wready[s];
```

Same idea for write data.

```systemverilog
        if (m_ar_target[m][s] && ar_grant[s][m])
          m_arready[m] = s_arready[s];
```

Same idea for read address.

At the end of section 4, the master sees ready only from the slave it is actively connected to.

## Section 5: Response Routing

This section routes `B` and `R` responses from slaves back to masters.

```systemverilog
  generate
    for (mi = 0; mi < NUM_MASTERS; mi++) begin : gen_responses
      always_comb begin
```

This creates one response-routing block per master.

Each block computes:

- what write response this master should see
- what read data this master should see

### Default outputs first

```systemverilog
        m_bvalid[mi] = 1'b0;
        m_bresp[mi]  = 2'b00;
        m_bid[mi]    = '0;
        m_rvalid[mi] = 1'b0;
        m_rdata[mi]  = '0;
        m_rresp[mi]  = 2'b00;
        m_rid[mi]    = '0;
        m_rlast[mi]  = 1'b0;
```

This resets the master's response view to "nothing happening".

Then the code searches through all slaves to see if one currently has a valid response for that master.

```systemverilog
        for (int s = 0; s < NUM_SLAVES; s++) begin
```

For every slave, ask:

- does the slave have a write response?
- is this master the owner of that slave's AW path?

or:

- does the slave have read data?
- is this master the owner of that slave's AR path?

```systemverilog
          if (s_bvalid[s] && aw_grant[s][mi]) begin
            m_bvalid[mi] = 1'b1;
            m_bresp[mi]  = s_bresp[s];
            m_bid[mi]    = s_bid[s];
          end
```

This means if slave `s` has a valid write response and master `mi` is the granted owner for that slave's write path, then:

- assert `m_bvalid[mi]`
- copy the slave's response code to `m_bresp[mi]`
- copy the slave's response ID to `m_bid[mi]`

So the write response is returned to the correct master.

```systemverilog
          if (s_rvalid[s] && ar_grant[s][mi]) begin
            m_rvalid[mi] = 1'b1;
            m_rdata[mi]  = s_rdata[s];
            m_rresp[mi]  = s_rresp[s];
            m_rid[mi]    = s_rid[s];
            m_rlast[mi]  = s_rlast[s];
          end
```

Same thing for read data.

If slave `s` has valid read data and master `mi` is the granted owner of that read path, then copy the slave's read-data channel back to that master.

At the end of section 5, each master sees only the responses that belong to it.

## Endmodule

```systemverilog
endmodule : axi_crossbar
```

This closes the module.

## One Full Worked Example

Suppose:

- master 0 wants to write to `0x1000_0040`
- master 1 wants to write to `0x4000_0100`

From the decoder mapping:

- address `0x1000_0040` targets slave 1
- address `0x4000_0100` targets slave 4

### Step 1: Decode

- `m_aw_target[0]` becomes one-hot for slave 1
- `m_aw_target[1]` becomes one-hot for slave 4

### Step 2: Build requests per slave

For slave 1:

- `aw_req = 2'b01`

For slave 4:

- `aw_req = 2'b10`

### Step 3: Arbitrate

- slave 1 grants master 0
- slave 4 grants master 1

So:

- `aw_grant[1] = 2'b01`
- `aw_grant[4] = 2'b10`

### Step 4: Mux forward

Slave 1 sees:

- `s_awvalid[1] = 1`
- `s_awaddr[1]  = m_awaddr[0]`

Slave 4 sees:

- `s_awvalid[4] = 1`
- `s_awaddr[4]  = m_awaddr[1]`

### Step 5: Ready back-propagation

- `m_awready[0] = s_awready[1]`
- `m_awready[1] = s_awready[4]`

### Step 6: Response routing

When slave 1 returns a B response, it goes to master 0.

When slave 4 returns a B response, it goes to master 1.

This is the main reason a crossbar is useful: different masters can talk to different slaves at the same time.

## One Important Architectural Note

This crossbar uses the current AW and AR grant information to route W, B, and R behavior.

That is reasonable for a simpler design with limited outstanding behavior, but it is not the full complexity of a large production AXI interconnect. A more advanced design would often keep explicit state about outstanding ownership across channels.

That is not necessarily a bug by itself here. It is just an important simplification to understand.

## Short Summary

This file is doing five jobs:

1. decode each master's address into a target slave
2. arbitrate independently at each slave
3. forward the winning master's AW, W, and AR signals to that slave
4. send ready back to the correct master
5. route B and R responses back to the correct master

If you want, the next useful follow-up is either:

- a separate markdown file that explains only the internal signals like `m_aw_target`, `aw_req`, `aw_grant`, and `aw_grant_idx`
- a cycle-by-cycle trace table for one write and one read through this exact crossbar