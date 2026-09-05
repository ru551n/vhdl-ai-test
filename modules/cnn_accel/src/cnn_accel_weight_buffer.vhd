library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

-- Double-buffered (ping-pong) on-chip weight/bias cache. See
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md and
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md.
--
-- Two banks (A = index 0, B = index 1), each holding a weight region
-- (rows of 'g_pe_rows*g_pe_cols' int8 lanes) and a bias region (rows of
-- 'g_pe_rows' 'g_accum_width'-bit lanes). 'fill_bank_sel' selects which
-- bank the AXI4-Stream fill port targets; 'fill_is_bias' selects which
-- region within that bank a fill beat targets (one lane per accepted
-- beat, auto-advancing a lane index and, once a row's lanes are all
-- written, a row/tile pointer). 'read_bank_sel' selects which bank the
-- two simple synchronous read ports (registered, 1 cycle latency) serve,
-- independently of the fill side -- the ping-pong property.
entity cnn_accel_weight_buffer is
  generic (
    -- Rows per bank, per region (weight region and bias region both use
    -- this same depth -- see proposal doc section 3.2).
    g_weight_buffer_depth : positive;
    -- Output-channel parallelism: rows per read tile, and bias lanes per
    -- read tile.
    g_pe_rows : positive;
    -- Input-channel/MAC parallelism: weight lanes per read tile, together
    -- with 'g_pe_rows'.
    g_pe_cols : positive;
    -- Bit width of one bias lane (int32 accumulator width elsewhere in
    -- this IP). Added by vhdesign to give 'bias_rd_data' a well-typed
    -- width -- see proposal doc section 3.1.
    g_accum_width : positive := 32
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- AXI4-Stream fill port, from the weight/bias 'cnn_accel_axi_read_dma'
    -- instance. One accepted beat writes one lane (one weight byte on
    -- 'data(7 downto 0)', or one bias lane on
    -- 'data(g_accum_width - 1 downto 0)') into the bank/region selected by
    -- 'fill_bank_sel'/'fill_is_bias'. 'last'/'user' are not used.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- Bank filled by 's_stream' ('0' = bank A, '1' = bank B).
    fill_bank_sel : in std_ulogic;
    -- '0' routes fill beats to the weight region, '1' to the bias region,
    -- of the bank selected by 'fill_bank_sel'.
    fill_is_bias : in std_ulogic;
    --# {{}}
    -- Bank served by 'weight_rd_addr'/'weight_rd_data' and
    -- 'bias_rd_addr'/'bias_rd_data' ('0' = bank A, '1' = bank B).
    read_bank_sel : in std_ulogic;
    --# {{}}
    -- Row (tile) address into the selected bank's weight region.
    weight_rd_addr : in std_ulogic_vector(num_bits_needed(g_weight_buffer_depth - 1) - 1 downto 0);
    -- One int8 weight per active PE ('g_pe_rows*g_pe_cols' lanes),
    -- registered, 1 cycle read latency.
    weight_rd_data : out std_ulogic_vector(8 * g_pe_rows * g_pe_cols - 1 downto 0);
    --# {{}}
    -- Row (tile) address into the selected bank's bias region.
    bias_rd_addr : in std_ulogic_vector(num_bits_needed(g_weight_buffer_depth - 1) - 1 downto 0);
    -- One int32 (g_accum_width-bit) bias per output channel lane
    -- ('g_pe_rows' lanes), registered, 1 cycle read latency.
    bias_rd_data : out std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0)
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

  -- Read-address width: enough to represent row indices 0 .. depth - 1.
  constant c_addr_width : positive := num_bits_needed(g_weight_buffer_depth - 1);
  -- Row-pointer width: enough to represent 0 .. depth *inclusive*, so the
  -- "reached depth" backpressure comparison never wraps around.
  constant c_ptr_width : positive := num_bits_needed(g_weight_buffer_depth);
  constant c_depth : unsigned(c_ptr_width - 1 downto 0) :=
    to_unsigned(g_weight_buffer_depth, c_ptr_width);

  constant c_weight_lane_width : positive := num_bits_needed(c_weight_lanes - 1);
  constant c_bias_lane_width : positive := num_bits_needed(c_bias_lanes - 1);

  ------------------------------------------------------------------------
  -- Bank memories: 2 banks (0 = A, 1 = B), each 'g_weight_buffer_depth'
  -- rows of one full read tile per region. Inferred simple-dual-port RAM
  -- idiom (hand-written array + one write process + one read process),
  -- following hdl-modules/modules/fifo/src/fifo.vhd's 'memory_block'
  -- pattern -- see proposal doc section 6.
  ------------------------------------------------------------------------

  type weight_row_t is array (0 to g_weight_buffer_depth - 1)
    of std_ulogic_vector(c_weight_row_width - 1 downto 0);
  type weight_bank_arr_t is array (0 to 1) of weight_row_t;
  signal weight_mem : weight_bank_arr_t;

  type bias_row_t is array (0 to g_weight_buffer_depth - 1)
    of std_ulogic_vector(c_bias_row_width - 1 downto 0);
  type bias_bank_arr_t is array (0 to 1) of bias_row_t;
  signal bias_mem : bias_bank_arr_t;

  ------------------------------------------------------------------------
  -- Per-bank fill state: one row pointer + lane index per region, per
  -- bank. Reset to 0 by 'reset' or by a 'fill_bank_sel' edge selecting
  -- that bank (a new fill session) -- see proposal doc section 3.4.
  ------------------------------------------------------------------------

  type ptr_arr_t is array (0 to 1) of unsigned(c_ptr_width - 1 downto 0);
  signal weight_wr_row_q : ptr_arr_t := (others => (others => '0'));
  signal bias_wr_row_q : ptr_arr_t := (others => (others => '0'));

  type weight_lane_arr_t is array (0 to 1) of unsigned(c_weight_lane_width - 1 downto 0);
  signal weight_lane_q : weight_lane_arr_t := (others => (others => '0'));

  type bias_lane_arr_t is array (0 to 1) of unsigned(c_bias_lane_width - 1 downto 0);
  signal bias_lane_q : bias_lane_arr_t := (others => (others => '0'));

  -- Previous-cycle 'fill_bank_sel', to detect a "new fill session" edge.
  signal fill_bank_sel_q : std_ulogic := '0';

  signal ready_i : std_ulogic;

  ------------------------------------------------------------------------
  function bank_index(sel : std_ulogic) return natural is
  begin
    if sel = '1' then
      return 1;
    else
      return 0;
    end if;
  end function;

begin

  ------------------------------------------------------------------------
  -- Backpressure: ready deasserts once the selected bank's (region-
  -- selected) row pointer has reached 'g_weight_buffer_depth'.
  ------------------------------------------------------------------------

  ready_i <=
    '0' when (fill_is_bias = '0' and weight_wr_row_q(bank_index(fill_bank_sel)) >= c_depth) else
    '0' when (fill_is_bias = '1' and bias_wr_row_q(bank_index(fill_bank_sel)) >= c_depth) else
    '1';

  s_stream_s2m.ready <= ready_i;

  ------------------------------------------------------------------------
  -- Fill path: one lane per accepted beat, into the bank/region selected
  -- by 'fill_bank_sel'/'fill_is_bias'.
  ------------------------------------------------------------------------

  fill : process(clk)
    variable bank : natural range 0 to 1;
    variable accepted : boolean;
  begin
    if rising_edge(clk) then
      bank := bank_index(fill_bank_sel);
      accepted := s_stream_m2s.valid = '1' and ready_i = '1';

      if reset then
        weight_wr_row_q <= (others => (others => '0'));
        bias_wr_row_q <= (others => (others => '0'));
        weight_lane_q <= (others => (others => '0'));
        bias_lane_q <= (others => (others => '0'));
        fill_bank_sel_q <= fill_bank_sel;
      else
        -- New fill session for the newly-selected bank: both regions'
        -- pointers/lane indices reset to 0 together.
        if fill_bank_sel /= fill_bank_sel_q then
          weight_wr_row_q(bank) <= (others => '0');
          bias_wr_row_q(bank) <= (others => '0');
          weight_lane_q(bank) <= (others => '0');
          bias_lane_q(bank) <= (others => '0');
        elsif accepted then
          if fill_is_bias = '0' then
            weight_mem(bank)(to_integer(weight_wr_row_q(bank)))
              (8 * (to_integer(weight_lane_q(bank)) + 1) - 1 downto 8 * to_integer(weight_lane_q(bank)))
              <= s_stream_m2s.data(7 downto 0);

            if to_integer(weight_lane_q(bank)) = c_weight_lanes - 1 then
              weight_lane_q(bank) <= (others => '0');
              weight_wr_row_q(bank) <= weight_wr_row_q(bank) + 1;
            else
              weight_lane_q(bank) <= weight_lane_q(bank) + 1;
            end if;
          else
            bias_mem(bank)(to_integer(bias_wr_row_q(bank)))
              (g_accum_width * (to_integer(bias_lane_q(bank)) + 1) - 1 downto g_accum_width * to_integer(bias_lane_q(bank)))
              <= s_stream_m2s.data(g_accum_width - 1 downto 0);

            if to_integer(bias_lane_q(bank)) = c_bias_lanes - 1 then
              bias_lane_q(bank) <= (others => '0');
              bias_wr_row_q(bank) <= bias_wr_row_q(bank) + 1;
            else
              bias_lane_q(bank) <= bias_lane_q(bank) + 1;
            end if;
          end if;
        end if;

        fill_bank_sel_q <= fill_bank_sel;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Read path: registered, 1 cycle latency, from the bank selected by
  -- 'read_bank_sel'. No reset -- read-data content has no completeness
  -- contract of its own (see proposal doc section 4).
  ------------------------------------------------------------------------

  read_ports : process(clk)
  begin
    if rising_edge(clk) then
      weight_rd_data <= weight_mem(bank_index(read_bank_sel))(to_integer(unsigned(weight_rd_addr)));
      bias_rd_data <= bias_mem(bank_index(read_bank_sel))(to_integer(unsigned(bias_rd_addr)));
    end if;
  end process;

end architecture a;
