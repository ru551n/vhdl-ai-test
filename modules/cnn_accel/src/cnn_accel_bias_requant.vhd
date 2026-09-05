library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;

-- Shared output-quantization stage. See
-- modules/cnn_accel/doc/cnn_accel_bias_requant_req.md and
-- modules/cnn_accel/doc/cnn_accel_bias_requant_proposal.md.
--
-- Per lane (one per output-channel PE row, 'g_pe_rows' lanes total), per
-- accepted 's_accum' beat:
--   total  = accum + (bias when cfg_bias_en else 0)
--   scaled = round_to_even(total * cfg_requant_scale, shift = 15 + cfg_requant_shift)
--            -- combined single rounding step: the Q15 fractional shift (15)
--            -- and cfg_requant_shift are folded into one shift amount, per
--            -- the requirement's documented option, so no double-rounding
--            -- error versus two separate rounding steps.
--   relu   = max(scaled, 0) when cfg_relu_en, applied BEFORE the int8 clamp
--   result = saturate_signed(relu, 8)
-- When cfg_requant_en='0': bias/relu are still applied to 'total', then the
-- low 8 bits of 'total' pass through as a two's-complement value (no
-- scaling, no saturation) -- debug/bypass path.
--
-- bias_rd_addr / bias tiling (v1 design decision, see proposal doc §3):
-- this module always drives 'bias_rd_addr' to all-zeros, i.e. it assumes a
-- single bias row (row 0) covers all 'g_pe_rows' output channels for the
-- whole layer (out_channels <= g_pe_rows). Multi-tile bias iteration would
-- need an explicit tile-index input not present on this module's port list
-- and is out of scope for v1.
entity cnn_accel_bias_requant is
  generic (
    -- Input accumulator width (int32 default).
    g_accum_width : positive := 32;
    -- Output-channel parallelism: number of independent lanes.
    g_pe_rows : positive := 8;
    -- Width of 'bias_rd_addr'. Added by vhdesign (not in the requirement)
    -- so this module's read-address port can be sized to match
    -- cnn_accel_weight_buffer's actual 'bias_rd_addr' width
    -- (num_bits_needed(g_weight_buffer_depth - 1)) at cnn_accel_top
    -- integration time. The value driven is always all-zeros in v1 (see
    -- entity-level comment above) regardless of this generic's value.
    g_bias_addr_width : positive := 1;
    -- Added by vhdesign: upper bound on the runtime-variable
    -- 'cfg_requant_shift' that this module supports at full precision.
    -- 'cfg_requant_shift' values above this bound are clamped to it (a
    -- defensive limit, not expected to be hit by compiled programs -- see
    -- proposal doc §4).
    g_max_requant_shift : natural := 31
  );
  port (
    clk : in std_ulogic;
    reset : in std_ulogic;

    cfg_bias_en : in std_ulogic;
    cfg_requant_en : in std_ulogic;
    cfg_relu_en : in std_ulogic;
    cfg_requant_scale : in std_ulogic_vector(31 downto 0);
    cfg_requant_shift : in std_ulogic_vector(7 downto 0);

    bias_rd_addr : out std_ulogic_vector(g_bias_addr_width - 1 downto 0);
    bias_rd_data : in std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0);

    s_accum_m2s : in axi_stream_m2s_t;
    s_accum_s2m : out axi_stream_s2m_t;

    m_out_m2s : out axi_stream_m2s_t;
    m_out_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_bias_requant;

architecture a of cnn_accel_bias_requant is

  -- One extra guard bit so 'accum + bias' cannot overflow (both operands
  -- are 'g_accum_width'-bit signed).
  constant c_sum_width : positive := g_accum_width + 1;
  -- Exact width of 'total * cfg_requant_scale' (signed(a) * signed(b) is
  -- exactly a+b bits wide, no truncation before rounding).
  constant c_product_width : positive := c_sum_width + 32;

  signal shift_amt_raw : natural range 0 to 255;
  signal shift_amt_clamped : natural range 0 to g_max_requant_shift;
  signal combined_shift : natural range 0 to 15 + g_max_requant_shift;

  signal scale_signed : signed(31 downto 0);

  signal m_out_data_low : std_ulogic_vector(8 * g_pe_rows - 1 downto 0);
  signal next_out_data_full : std_ulogic_vector(axi_stream_data_sz - 1 downto 0);

  signal accept_input : std_ulogic;
  signal out_valid_q : std_ulogic := '0';
  signal out_data_q : std_ulogic_vector(axi_stream_data_sz - 1 downto 0) := (others => '0');
  signal out_last_q : std_ulogic := '0';

  -- Round 'product_in' to the nearest integer after an arithmetic right
  -- shift by 'shift_amt' bits (runtime-variable), ties rounding to even.
  -- Deliberately hand-written rather than instantiating
  -- 'math.truncate_round_signed': that entity's number of removed LSBs is
  -- fixed by its 'input_width'/'result_width' generics at elaboration time,
  -- but 'shift_amt' here is a genuine runtime value derived from
  -- 'cfg_requant_shift' (an ISA field, loaded per instruction) -- see
  -- proposal doc §4 for the full rationale (a per-shift-value generate/mux
  -- array was considered and rejected: resource-explosive, and *not*
  -- reusing it would otherwise force two separate rounding steps, which
  -- would double-round versus the golden model's single combined shift).
  -- Structurally mirrors 'truncate_round_signed's algorithm (compare twice
  -- the remainder against the divisor, tie -> round to even), verified
  -- bit-exact against cnn_accel_model.py's 'round_shift_right_signed' by
  -- the testbench.
  function round_shift_right(
    product_in : signed(c_product_width - 1 downto 0);
    shift_amt : natural
  ) return signed is
    variable quotient : signed(c_product_width - 1 downto 0);
    variable quotient_shifted_back : signed(c_product_width - 1 downto 0);
    variable remainder : signed(c_product_width - 1 downto 0);
    variable remainder_ext : signed(c_product_width downto 0);
    variable divisor_ext : signed(c_product_width downto 0);
    variable twice_remainder : signed(c_product_width downto 0);
    variable result : signed(c_product_width - 1 downto 0);
  begin
    quotient := shift_right(product_in, shift_amt);
    quotient_shifted_back := shift_left(quotient, shift_amt);
    remainder := product_in - quotient_shifted_back;

    remainder_ext := resize(remainder, remainder_ext'length);
    twice_remainder := shift_left(remainder_ext, 1);
    divisor_ext := shift_left(to_signed(1, divisor_ext'length), shift_amt);

    if twice_remainder < divisor_ext then
      result := quotient;
    elsif twice_remainder > divisor_ext then
      result := quotient + 1;
    else
      -- Exact tie: round to even (quotient's LSB is its parity bit).
      if quotient(0) = '0' then
        result := quotient;
      else
        result := quotient + 1;
      end if;
    end if;

    return result;
  end function;

begin

  assert g_accum_width * g_pe_rows <= axi_stream_data_sz
    report "cnn_accel_bias_requant: g_accum_width*g_pe_rows (" &
      natural'image(g_accum_width * g_pe_rows) &
      ") exceeds axi_stream_pkg's fixed data width (" &
      natural'image(axi_stream_data_sz) & ") -- see proposal doc §2.1"
    severity failure;

  assert 8 * g_pe_rows <= axi_stream_data_sz
    report "cnn_accel_bias_requant: 8*g_pe_rows exceeds axi_stream_pkg's fixed data width"
    severity failure;

  assert g_accum_width >= 8
    report "cnn_accel_bias_requant: g_accum_width must be >= 8 (bypass path needs 8 low bits)"
    severity failure;

  assert (15 + g_max_requant_shift) <= (c_product_width - 2)
    report "cnn_accel_bias_requant: g_max_requant_shift too large for g_accum_width"
    severity failure;

  ------------------------------------------------------------------------
  -- Shared (per-beat, all lanes) control signal decoding.
  ------------------------------------------------------------------------

  scale_signed <= signed(cfg_requant_scale);

  shift_amt_raw <= to_integer(unsigned(cfg_requant_shift));
  shift_amt_clamped <= shift_amt_raw when shift_amt_raw <= g_max_requant_shift else g_max_requant_shift;
  combined_shift <= 15 + shift_amt_clamped;

  ------------------------------------------------------------------------
  -- v1 bias addressing: single bias row (row 0) for the whole layer --
  -- see entity-level comment.
  ------------------------------------------------------------------------

  bias_rd_addr <= (others => '0');

  ------------------------------------------------------------------------
  -- Per-lane datapath.
  ------------------------------------------------------------------------

  lane_gen : for l in 0 to g_pe_rows - 1 generate

    signal accum_l : signed(g_accum_width - 1 downto 0);
    signal bias_l : signed(g_accum_width - 1 downto 0);
    signal total_l : signed(c_sum_width - 1 downto 0);
    signal product_l : signed(c_product_width - 1 downto 0);
    signal scaled_l : signed(c_product_width - 1 downto 0);
    signal relu_scaled_l : signed(c_product_width - 1 downto 0);
    signal sat_result_l : signed(7 downto 0);
    signal bypass_result_l : signed(7 downto 0);
    signal final_lane_l : signed(7 downto 0);

  begin

    accum_l <= signed(s_accum_m2s.data(g_accum_width * (l + 1) - 1 downto g_accum_width * l));
    bias_l <= signed(bias_rd_data(g_accum_width * (l + 1) - 1 downto g_accum_width * l));

    total_l <= resize(accum_l, c_sum_width) + resize(bias_l, c_sum_width) when cfg_bias_en = '1' else
               resize(accum_l, c_sum_width);

    -- Requant path: total * scale (Q15), rounded shift, ReLU-before-saturate,
    -- saturate to int8 (genuine reuse of 'math.saturate_signed').
    product_l <= total_l * scale_signed;

    scaled_l <= round_shift_right(product_l, combined_shift);

    relu_scaled_l <= (others => '0') when (cfg_relu_en = '1' and scaled_l(scaled_l'high) = '1') else
                     scaled_l;

    saturate_signed_inst : entity math.saturate_signed
      generic map (
        input_width => c_product_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => relu_scaled_l,
        result_valid => open,
        result_value => sat_result_l,
        result_is_saturated => open
      );

    -- Bypass path (cfg_requant_en='0'): bias/ReLU still applied to 'total',
    -- then the low 8 bits pass through as two's-complement (no scaling, no
    -- saturation).
    bypass_proc : process(all)
      variable relu_clamped_total : signed(c_sum_width - 1 downto 0);
      -- Intermediate variable needed: some tools (e.g. nvc) reject slicing
      -- a type conversion's result directly ("the prefix of a slice name
      -- must be a name or a function call"), so the conversion is bound to
      -- a variable first and that variable is sliced.
      variable relu_clamped_slv : std_ulogic_vector(c_sum_width - 1 downto 0);
    begin
      if cfg_relu_en = '1' and total_l(total_l'high) = '1' then
        relu_clamped_total := (others => '0');
      else
        relu_clamped_total := total_l;
      end if;
      relu_clamped_slv := std_ulogic_vector(relu_clamped_total);
      bypass_result_l <= signed(relu_clamped_slv(7 downto 0));
    end process;

    final_lane_l <= sat_result_l when cfg_requant_en = '1' else bypass_result_l;

    m_out_data_low(8 * (l + 1) - 1 downto 8 * l) <= std_ulogic_vector(final_lane_l);

  end generate lane_gen;

  ------------------------------------------------------------------------
  -- Output packing and single-stage flow-through register: registers
  -- valid/data/last with reset, but never inserts a bubble (accepts a new
  -- input beat in the same cycle it drains the current one to a ready
  -- consumer) -- full throughput, one cycle latency. Reset clears
  -- 'out_valid_q' only (no completeness contract on stale data content).
  ------------------------------------------------------------------------

  next_out_data_full(8 * g_pe_rows - 1 downto 0) <= m_out_data_low;
  data_padding_gen : if 8 * g_pe_rows < axi_stream_data_sz generate
    next_out_data_full(axi_stream_data_sz - 1 downto 8 * g_pe_rows) <= (others => '0');
  end generate;

  accept_input <= (not out_valid_q) or m_out_s2m.ready;
  s_accum_s2m.ready <= accept_input;

  output_register : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        out_valid_q <= '0';
      else
        if accept_input = '1' then
          out_valid_q <= s_accum_m2s.valid;
          out_data_q <= next_out_data_full;
          out_last_q <= s_accum_m2s.last;
        end if;
      end if;
    end if;
  end process;

  m_out_m2s.valid <= out_valid_q;
  m_out_m2s.data <= out_data_q;
  m_out_m2s.last <= out_last_q;
  m_out_m2s.user <= (others => '0');

end architecture a;
