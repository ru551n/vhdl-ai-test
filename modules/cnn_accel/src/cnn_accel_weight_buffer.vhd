library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

library fifo;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Single-buffered on-chip weight/bias cache. See
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md and
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md.
--
-- Weights are streamed from DDR4 by a (future) DMA, one output-channel
-- pass at a time: 'fill_start' pulses once to reset the fill write
-- pointers, the whole weight set (weight region, then/interleaved with the
-- bias region, selected by 'fill_is_bias') is streamed in over 's_stream',
-- the pass runs (reading through 'weight_rd_addr'/'bias_rd_addr'), then the
-- next pass's 'fill_start' pulse begins the next refill. There is no
-- second bank to hide the refill behind compute any more -- an optional
-- shallow prefetch FIFO on the fill stream ('g_fill_fifo_depth') is the
-- mechanism that absorbs DDR4/DMA burst latency instead.
--
-- One weight region (rows of 'g_pe_rows*g_pe_cols' int8 lanes,
-- 'g_weight_buffer_depth' rows) and one, independently-sized, bias region
-- (rows of 'g_pe_rows' 'g_accum_width'-bit lanes, 'g_bias_buffer_depth'
-- rows -- far shallower, since a real layer only ever needs
-- 'ceil(out_channels/g_pe_rows)' bias rows). Fill beats arrive one int8/
-- int32 lane at a time; lanes are assembled into a row-wide register and
-- committed to memory with a single wide write once a row's last lane
-- arrives (see the fill process below) -- this, not the double buffering,
-- is what lets Yosys infer one wide block RAM per region instead of
-- splitting the weight memory into one RAMB18 per lane.
--
-- ISA v1.2 (H2, doc/tosa_compiler_plan.md section 5 extension 2): a third,
-- scale region ('fill_is_scale') holds the per-channel requant table --
-- rows of 'g_pe_rows' 'c_scale_entry_width' (40)-bit lanes, one row per
-- output-channel tile, the SAME depth and read address ('bias_rd_addr') as
-- the bias region, because the two tables are tiled identically
-- (cnn_accel_model.pack_bias_for_hw / pack_scale_table_for_hw, entry
-- 'ot*g_pe_rows + r' -> row 'ot' lane 'r') and are consumed together per
-- beat by cnn_accel_bias_requant. It is filled through the same fill
-- stream in the same tile-load phase as the bias region, one 8-byte DDR
-- entry ('data(39 downto 0)' = multiplier i32 LE, shift u8) per beat. A
-- design that leaves 'fill_is_scale' at '0' never writes the region and
-- is bit-identical to the pre-H2 buffer on every other port.
entity cnn_accel_weight_buffer is
  generic (
    -- Rows in the weight region.
    g_weight_buffer_depth : positive;
    -- Rows in the (separate, much shallower) bias region -- independent of
    -- 'g_weight_buffer_depth' so it can fall out of block RAM into
    -- LUTRAM/registers (see proposal doc section 3.2).
    g_bias_buffer_depth : positive := 8;
    -- Output-channel parallelism: rows per read tile, and bias lanes per
    -- read tile.
    g_pe_rows : positive;
    -- Input-channel/MAC parallelism: weight lanes per read tile, together
    -- with 'g_pe_rows'.
    g_pe_cols : positive;
    -- Bit width of one bias lane (int32 accumulator width elsewhere in
    -- this IP). Added by vhdesign to give 'bias_rd_data' a well-typed
    -- width -- see proposal doc section 3.1.
    g_accum_width : positive := 32;
    -- Depth of the shallow prefetch FIFO placed on the fill stream, ahead
    -- of the row-assembly/write logic, to absorb DDR4/DMA burst latency.
    -- '0' means no FIFO is instantiated at all (the fill stream connects
    -- straight through, exactly the old single-bank timing). Reuses
    -- hdl-modules' 'fifo.fifo' (shared/ReusableRTL.md) -- must be a power
    -- of two whenever nonzero (that entity's own constraint).
    g_fill_fifo_depth : natural := 32
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- AXI4-Stream fill port, from the weight/bias 'cnn_accel_axi_read_dma'
    -- instance (optionally through the internal prefetch FIFO). One
    -- accepted beat carries one lane (one weight byte on
    -- 'data(7 downto 0)', or one bias lane on
    -- 'data(g_accum_width - 1 downto 0)') into the region selected by
    -- 'fill_is_bias'. 'last'/'user' are not used.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- Pulse: starts a new fill session -- resets the weight and bias
    -- write row pointers/lane indices/row-assembly registers to 0. Must be
    -- pulsed once before streaming a new output-channel pass's weight/bias
    -- set (replaces the old 'fill_bank_sel'-edge-triggered "new fill
    -- session" detector -- see proposal doc section 3.4). A beat presented
    -- on the same cycle as 'fill_start' is not accepted.
    fill_start : in std_ulogic := '0';
    -- '0' routes fill beats to the weight region, '1' to the bias region
    -- (unless 'fill_is_scale' is '1').
    fill_is_bias : in std_ulogic;
    -- '1' routes fill beats to the per-channel scale region (ISA v1.2),
    -- overriding 'fill_is_bias'. One 'c_scale_entry_width'-bit lane
    -- ('data(c_scale_entry_width - 1 downto 0)') per accepted beat.
    fill_is_scale : in std_ulogic := '0';
    --# {{}}
    -- Row (tile) address into the weight region.
    weight_rd_addr : in std_ulogic_vector(num_bits_needed(g_weight_buffer_depth - 1) - 1 downto 0);
    -- Clock enable for the WEIGHT region's two-stage read pipeline (and
    -- only that -- the bias/scale reads are unaffected). '0' freezes both
    -- stages, so 'weight_rd_data' is bit-identical on a frozen cycle and
    -- the reader's own frozen pipeline stays paired with it. Defaults to
    -- '1', i.e. the always-on read this port used to be.
    weight_rd_en : in std_ulogic := '1';
    -- One int8 weight per active PE ('g_pe_rows*g_pe_cols' lanes),
    -- registered, **2 cycle** read latency (see the block-RAM output
    -- register note on the read process below), advanced only on cycles
    -- where 'weight_rd_en' is '1'. The bias and scale ports below still
    -- have 1 cycle of latency and no enable.
    weight_rd_data : out std_ulogic_vector(8 * g_pe_rows * g_pe_cols - 1 downto 0);
    --# {{}}
    -- Row (tile) address into the bias region.
    bias_rd_addr : in std_ulogic_vector(num_bits_needed(g_bias_buffer_depth - 1) - 1 downto 0);
    -- One int32 (g_accum_width-bit) bias per output channel lane
    -- ('g_pe_rows' lanes), registered, 1 cycle read latency.
    bias_rd_data : out std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0);
    -- One per-channel requant table entry (multiplier + shift, see
    -- cnn_accel_pkg's 'c_scale_entry_width' comment) per output channel
    -- lane of the row addressed by 'bias_rd_addr', registered, 1 cycle
    -- read latency -- same timing as 'bias_rd_data'.
    scale_rd_data : out std_ulogic_vector(c_scale_entry_width * g_pe_rows - 1 downto 0)
  );
end entity cnn_accel_weight_buffer;

architecture a of cnn_accel_weight_buffer is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_weight_lanes : positive := g_pe_rows * g_pe_cols;
  constant c_bias_lanes : positive := g_pe_rows;

  constant c_weight_row_width : positive := 8 * c_weight_lanes;
  constant c_bias_row_width : positive := g_accum_width * c_bias_lanes;
  -- Scale region: same lane count, row count and read address as the
  -- bias region (entity-level comment).
  constant c_scale_row_width : positive := c_scale_entry_width * c_bias_lanes;

  -- Read-address width: enough to represent row indices 0 .. depth - 1.
  constant c_weight_addr_width : positive := num_bits_needed(g_weight_buffer_depth - 1);
  constant c_bias_addr_width : positive := num_bits_needed(g_bias_buffer_depth - 1);
  -- Row-pointer width: enough to represent 0 .. depth *inclusive*, so the
  -- "reached depth" backpressure comparison never wraps around.
  constant c_weight_ptr_width : positive := num_bits_needed(g_weight_buffer_depth);
  constant c_bias_ptr_width : positive := num_bits_needed(g_bias_buffer_depth);
  constant c_weight_depth : unsigned(c_weight_ptr_width - 1 downto 0) :=
    to_unsigned(g_weight_buffer_depth, c_weight_ptr_width);
  constant c_bias_depth : unsigned(c_bias_ptr_width - 1 downto 0) :=
    to_unsigned(g_bias_buffer_depth, c_bias_ptr_width);

  constant c_weight_lane_width : positive := num_bits_needed(c_weight_lanes - 1);
  constant c_bias_lane_width : positive := num_bits_needed(c_bias_lanes - 1);

  -- Number of fill-stream payload bits actually consumed (a weight lane
  -- needs 8 bits, a bias lane needs 'g_accum_width', a scale lane
  -- 'c_scale_entry_width' -- the widest of the three is what has to
  -- survive a trip through the optional prefetch FIFO).
  function max_pos(a, b : positive) return positive is
  begin
    if a > b then
      return a;
    else
      return b;
    end if;
  end function;

  constant c_payload_width : positive := max_pos(max_pos(8, g_accum_width), c_scale_entry_width);
  -- Payload bits, plus two bits carrying 'fill_is_bias'/'fill_is_scale'
  -- alongside the data through the FIFO so a beat is always routed to the
  -- region it was destined for at accept time, regardless of any region-
  -- select change while earlier beats are still buffered.
  constant c_fifo_width : positive := c_payload_width + 2;

  ------------------------------------------------------------------------
  -- Region memories: one weight region, one (independently-sized) bias
  -- region. Inferred simple-dual-port RAM idiom (hand-written array + one
  -- write process + one read process), following
  -- hdl-modules/modules/fifo/src/fifo.vhd's 'memory_block' pattern -- see
  -- proposal doc section 6. Each region is written with ONE wide row write
  -- per completed row (row-assembly register below), not a per-lane
  -- decoded byte-write-enable loop -- this is what lets Yosys infer one
  -- block RAM per region instead of one RAMB18 per lane.
  ------------------------------------------------------------------------

  type weight_mem_t is array (0 to g_weight_buffer_depth - 1)
    of std_ulogic_vector(c_weight_row_width - 1 downto 0);
  signal weight_mem : weight_mem_t;

  type bias_mem_t is array (0 to g_bias_buffer_depth - 1)
    of std_ulogic_vector(c_bias_row_width - 1 downto 0);
  signal bias_mem : bias_mem_t;

  type scale_mem_t is array (0 to g_bias_buffer_depth - 1)
    of std_ulogic_vector(c_scale_row_width - 1 downto 0);
  signal scale_mem : scale_mem_t;

  ------------------------------------------------------------------------
  -- Fill state: one row pointer + lane index + row-assembly register per
  -- region. Reset to 0/empty by 'reset' or by a 'fill_start' pulse (a new
  -- fill session) -- see proposal doc section 3.4.
  ------------------------------------------------------------------------

  signal weight_wr_row_q : unsigned(c_weight_ptr_width - 1 downto 0) := (others => '0');
  signal weight_lane_q : unsigned(c_weight_lane_width - 1 downto 0) := (others => '0');
  signal weight_row_assemble_q : std_ulogic_vector(c_weight_row_width - 1 downto 0) :=
    (others => '0');

  signal bias_wr_row_q : unsigned(c_bias_ptr_width - 1 downto 0) := (others => '0');
  signal bias_lane_q : unsigned(c_bias_lane_width - 1 downto 0) := (others => '0');
  signal bias_row_assemble_q : std_ulogic_vector(c_bias_row_width - 1 downto 0) :=
    (others => '0');

  signal scale_wr_row_q : unsigned(c_bias_ptr_width - 1 downto 0) := (others => '0');
  signal scale_lane_q : unsigned(c_bias_lane_width - 1 downto 0) := (others => '0');
  signal scale_row_assemble_q : std_ulogic_vector(c_scale_row_width - 1 downto 0) :=
    (others => '0');

  signal ready_i : std_ulogic;

  -- The weight region's block-RAM DO stage; 'weight_rd_data' is its
  -- output register. See the read process below.
  signal weight_rd_data_p : std_ulogic_vector(8 * g_pe_rows * g_pe_cols - 1 downto 0) :=
    (others => '0');

  ------------------------------------------------------------------------
  -- Internal (post-FIFO, or straight-through when 'g_fill_fifo_depth=0')
  -- fill-accept signals: a beat's data and the 'fill_is_bias' value it was
  -- accepted with, bundled together so region routing is always correct
  -- even when a beat has been sitting in the prefetch FIFO across a
  -- 'fill_is_bias' change.
  ------------------------------------------------------------------------

  signal valid_i : std_ulogic;
  signal data_i : std_ulogic_vector(c_payload_width - 1 downto 0);
  signal is_bias_i : std_ulogic;
  signal is_scale_i : std_ulogic;

  signal fifo_write_valid : std_ulogic;
  signal fifo_write_ready : std_ulogic;
  signal fifo_write_data : std_ulogic_vector(c_fifo_width - 1 downto 0);
  signal fifo_read_valid : std_ulogic;
  signal fifo_read_ready : std_ulogic;
  signal fifo_read_data : std_ulogic_vector(c_fifo_width - 1 downto 0);

begin

  ------------------------------------------------------------------------
  -- Backpressure: ready deasserts once the selected region's row pointer
  -- has reached its own depth.
  ------------------------------------------------------------------------

  ready_i <=
    '0' when (is_scale_i = '1' and scale_wr_row_q >= c_bias_depth) else
    '0' when (is_scale_i = '0' and is_bias_i = '0' and weight_wr_row_q >= c_weight_depth) else
    '0' when (is_scale_i = '0' and is_bias_i = '1' and bias_wr_row_q >= c_bias_depth) else
    '1';

  ------------------------------------------------------------------------
  -- Optional shallow prefetch FIFO on the fill stream (proposal doc
  -- section on the prefetch FIFO / doc/cnn_accel_weight_buffer.md):
  -- absorbs DDR4/DMA burst latency now that there is no second bank to
  -- hide it behind. Reuses hdl-modules' 'fifo.fifo' unmodified
  -- (shared/ReusableRTL.md) rather than a hand-rolled memory.
  ------------------------------------------------------------------------

  no_fifo_gen : if g_fill_fifo_depth = 0 generate
    valid_i <= s_stream_m2s.valid;
    data_i <= s_stream_m2s.data(c_payload_width - 1 downto 0);
    is_bias_i <= fill_is_bias;
    is_scale_i <= fill_is_scale;
    s_stream_s2m.ready <= ready_i;
  end generate;

  fill_fifo_gen : if g_fill_fifo_depth > 0 generate
    fifo_write_valid <= s_stream_m2s.valid;
    fifo_write_data <= fill_is_scale & fill_is_bias & s_stream_m2s.data(c_payload_width - 1 downto 0);
    s_stream_s2m.ready <= fifo_write_ready;

    valid_i <= fifo_read_valid;
    data_i <= fifo_read_data(c_payload_width - 1 downto 0);
    is_bias_i <= fifo_read_data(c_payload_width);
    is_scale_i <= fifo_read_data(c_payload_width + 1);
    fifo_read_ready <= ready_i;

    fill_fifo_inst : entity fifo.fifo
      generic map (
        width => c_fifo_width,
        -- '+ 1' because of 'enable_output_register' below: 'fifo.fifo'
        -- takes one word of 'depth' for the output register itself
        -- ('memory_depth := depth - 1') and then asserts that what is left
        -- is a power of two. Passing 'g_fill_fifo_depth' unchanged makes
        -- the RAM 31 deep and trips that assert. With the '+ 1' the RAM is
        -- exactly the 'g_fill_fifo_depth' words it always was, and the
        -- FIFO's usable capacity is one word MORE than before (the word
        -- sitting in the output register), never less -- so no fill
        -- sequence that fitted before can fail to fit now.
        depth => g_fill_fifo_depth + 1,
        -- Block-RAM OUTPUT REGISTER on the prefetch FIFO (shared/
        -- TimingAndResources.md, "Memories and lookup": use the block
        -- RAM's output register; never put logic between a RAM's data
        -- output and the first register).
        --
        -- Without it 'fifo_read_data' is the FIFO memory's raw DO port,
        -- and 'data_i' -- the lane word taken straight off it -- runs
        -- combinationally through the row-assembly lane mux into
        -- 'bias_mem'/'weight_mem'/'scale_mem''s DI pins. Post-route that
        -- was 'fill_fifo/mem_reg/CLKARDCLK -> bias_mem_reg_0/DIADI[1]' at
        -- -1.273 ns with ZERO logic levels of margin to recover: ~2.1 ns
        -- of RAMB36 clock-to-out plus the route to another block RAM's
        -- data-input pins, which is precisely the case the rule names.
        -- With the output register the FIFO's read data leaves a
        -- flip-flop instead, at a fraction of the clock-to-out.
        --
        -- Free here: the FIFO is a handshake ('fifo.fifo' keeps
        -- 'read_valid'/'read_ready' coherent with the extra stage itself,
        -- and reserves one word of the same 'depth' for it), so this
        -- costs one cycle of fill LATENCY and no throughput at all -- the
        -- fill stream still accepts one lane per cycle. Nothing outside
        -- this entity can observe the difference: the fill port is a
        -- ready/valid stream with no cycle contract.
        enable_output_register => true
      )
      port map (
        clk => clk,
        write_ready => fifo_write_ready,
        write_valid => fifo_write_valid,
        write_data => fifo_write_data,
        read_ready => fifo_read_ready,
        read_valid => fifo_read_valid,
        read_data => fifo_read_data
      );
  end generate;

  ------------------------------------------------------------------------
  -- Fill path: one lane per accepted beat, assembled into a row-wide
  -- register; the region's memory only receives ONE wide write per row,
  -- issued when the row's final lane arrives.
  ------------------------------------------------------------------------

  fill : process(clk)
    variable accepted : boolean;
    variable weight_row_next : std_ulogic_vector(c_weight_row_width - 1 downto 0);
    variable bias_row_next : std_ulogic_vector(c_bias_row_width - 1 downto 0);
    variable scale_row_next : std_ulogic_vector(c_scale_row_width - 1 downto 0);
  begin
    if rising_edge(clk) then
      accepted := valid_i = '1' and ready_i = '1';

      if reset then
        weight_wr_row_q <= (others => '0');
        weight_lane_q <= (others => '0');
        weight_row_assemble_q <= (others => '0');
        bias_wr_row_q <= (others => '0');
        bias_lane_q <= (others => '0');
        bias_row_assemble_q <= (others => '0');
        scale_wr_row_q <= (others => '0');
        scale_lane_q <= (others => '0');
        scale_row_assemble_q <= (others => '0');
      elsif fill_start = '1' then
        -- New fill session: all regions' pointers/lane indices/row-
        -- assembly registers reset to 0 together (a beat presented this
        -- same cycle is not accepted).
        weight_wr_row_q <= (others => '0');
        weight_lane_q <= (others => '0');
        weight_row_assemble_q <= (others => '0');
        bias_wr_row_q <= (others => '0');
        bias_lane_q <= (others => '0');
        bias_row_assemble_q <= (others => '0');
        scale_wr_row_q <= (others => '0');
        scale_lane_q <= (others => '0');
        scale_row_assemble_q <= (others => '0');
      elsif accepted then
        if is_scale_i = '1' then
          -- Scale region (ISA v1.2): constant-bound loop, same reason as
          -- the weight region below.
          scale_row_next := scale_row_assemble_q;
          for lane in 0 to c_bias_lanes - 1 loop
            if to_integer(scale_lane_q) = lane then
              scale_row_next(c_scale_entry_width * (lane + 1) - 1 downto c_scale_entry_width * lane) :=
                data_i(c_scale_entry_width - 1 downto 0);
            end if;
          end loop;
          scale_row_assemble_q <= scale_row_next;

          if to_integer(scale_lane_q) = c_bias_lanes - 1 then
            -- Final lane of the row: ONE wide write, all lanes at once.
            scale_mem(to_integer(scale_wr_row_q)) <= scale_row_next;
            scale_lane_q <= (others => '0');
            scale_wr_row_q <= scale_wr_row_q + 1;
          else
            scale_lane_q <= scale_lane_q + 1;
          end if;
        elsif is_bias_i = '0' then
          -- Constant-bound loop with the lane select as a per-lane
          -- enable, rather than a dynamically-bounded slice
          -- '(8*(to_integer(lane)+1)-1 downto 8*to_integer(lane))'.
          -- Identical in simulation, but GHDL's synthesis backend rejects
          -- the latter ("cannot extract same variable part for dynamic
          -- slice", the same limitation as ghdl/ghdl#2658). This only
          -- updates a plain register (not a memory), so it costs no BRAM
          -- fragmentation either way -- the actual memory write below is
          -- one whole-row write, no per-lane enables at all.
          weight_row_next := weight_row_assemble_q;
          for lane in 0 to c_weight_lanes - 1 loop
            if to_integer(weight_lane_q) = lane then
              weight_row_next(8 * (lane + 1) - 1 downto 8 * lane) := data_i(7 downto 0);
            end if;
          end loop;
          weight_row_assemble_q <= weight_row_next;

          if to_integer(weight_lane_q) = c_weight_lanes - 1 then
            -- Final lane of the row: ONE wide write, all lanes at once.
            weight_mem(to_integer(weight_wr_row_q)) <= weight_row_next;
            weight_lane_q <= (others => '0');
            weight_wr_row_q <= weight_wr_row_q + 1;
          else
            weight_lane_q <= weight_lane_q + 1;
          end if;
        else
          -- Constant-bound loop, same reason as the weight region above.
          bias_row_next := bias_row_assemble_q;
          for lane in 0 to c_bias_lanes - 1 loop
            if to_integer(bias_lane_q) = lane then
              bias_row_next(g_accum_width * (lane + 1) - 1 downto g_accum_width * lane) :=
                data_i(g_accum_width - 1 downto 0);
            end if;
          end loop;
          bias_row_assemble_q <= bias_row_next;

          if to_integer(bias_lane_q) = c_bias_lanes - 1 then
            -- Final lane of the row: ONE wide write, all lanes at once.
            bias_mem(to_integer(bias_wr_row_q)) <= bias_row_next;
            bias_lane_q <= (others => '0');
            bias_wr_row_q <= bias_wr_row_q + 1;
          else
            bias_lane_q <= bias_lane_q + 1;
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Read path. No reset -- read-data content has no completeness contract
  -- of its own (see proposal doc section 4).
  --
  -- BLOCK-RAM OUTPUT REGISTER on the weight region (shared/
  -- TimingAndResources.md, "Memories and lookup": use the block RAM's
  -- output register; it costs one cycle and removes the RAM's clock-to-out
  -- plus routing from the following stage).
  --
  -- 'weight_rd_data' feeds cnn_accel_pe_array's DSP48 A/B ports directly,
  -- with no logic in between, and post-route that single-register read was
  -- the largest group of failing endpoints in the whole accelerator:
  --
  --   'weight_buffer/weight_mem_reg_1/CLKARDCLK ->
  --    pe_array/dsp_q_reg[1][0]/D[16]'   -0.985 ns, ZERO logic levels,
  --    3.444 ns of data path of which 2.125 ns is RAMB36 clock-to-out
  --
  -- 388 of the 400 worst paths in the build had exactly that shape. With
  -- nothing in the path to restructure, the clock-to-out itself is the
  -- only term left, and the primitive's own output register is what
  -- removes it: 'weight_rd_data_p' below is the RAM's DO port and
  -- 'weight_rd_data' the DOREG stage, so Vivado maps the pair onto one
  -- RAMB36E1 with 'DO*_REG = 1' rather than a RAMB plus fabric
  -- flip-flops.
  --
  -- The extra cycle is absorbed by cnn_accel_pe_array's tap pipeline
  -- ('tap2_q' there), so weights and activations still meet in the same
  -- multiply and the array's throughput is unchanged. 'weight_rd_en'
  -- exists for the other half of that contract: pe_array freezes whole
  -- when its output register is full, and a two-stage read with no enable
  -- would keep advancing and hand the frozen multiply the NEXT group's
  -- weights. (With the old single-stage read, re-presenting the same
  -- address was enough; with two stages it no longer is, because the
  -- pipeline would still be draining the row behind it.)
  --
  -- Bias and scale are deliberately left at one cycle: they feed
  -- cnn_accel_bias_requant, whose own pipeline expects that, and they were
  -- nowhere near the critical path.
  ------------------------------------------------------------------------

  read_ports : process(clk)
  begin
    if rising_edge(clk) then
      if weight_rd_en = '1' then
        weight_rd_data_p <= weight_mem(to_integer(unsigned(weight_rd_addr)));
        weight_rd_data <= weight_rd_data_p;
      end if;
      bias_rd_data <= bias_mem(to_integer(unsigned(bias_rd_addr)));
      scale_rd_data <= scale_mem(to_integer(unsigned(bias_rd_addr)));
    end if;
  end process;

end architecture a;
