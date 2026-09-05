library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Spatial reduction over a 'cnn_accel_window_gen'-produced pooling window.
-- See modules/cnn_accel/doc/cnn_accel_pool_req.md and
-- modules/cnn_accel/doc/cnn_accel_pool_proposal.md.
--
-- 'OPCODE_POOL_MAX': int8 max over the window's active taps, emitted
-- directly on 'm_max' (bypasses 'cnn_accel_bias_requant'). 'OPCODE_POOL_AVG':
-- int32 (g_accum_width-bit) sum over the window's active taps, emitted on
-- 'm_avgsum' for 'cnn_accel_bias_requant' to scale/round (division by the
-- pool area is a downstream requantize op -- see doc/cnn_accel_arch.md
-- "Non-obvious boundary rationale"). The two outputs are structurally
-- mutually exclusive: one shared output register, tagged with which port
-- it belongs to -- see proposal doc section 4.
entity cnn_accel_pool is
  generic (
    -- Upper bound on 'cfg_pool_kernel_h'/'cfg_pool_kernel_w' individually;
    -- sizes the fixed 'g_max_kernel_size**2'-lane tap array/reduction
    -- network. Contract: 'g_max_kernel_size**2 * 8 <= axi_stream_data_sz'
    -- (128), since 's_window_m2s.data' uses the fixed-width
    -- 'axi_stream_pkg' record type -- see proposal doc section 2.
    g_max_kernel_size : positive;
    -- 'OPCODE_POOL_AVG' sum width. Must be wide enough to hold
    -- 'g_max_kernel_size**2 * 127' (the maximum possible window sum)
    -- without overflow -- no saturation/rounding is applied in this
    -- module, per doc/cnn_accel_pool_req.md.
    g_accum_width : positive
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- 'OPCODE_POOL_MAX' vs 'OPCODE_POOL_AVG' (cnn_accel_pkg), sampled at
    -- 's_window' accept time -- selects whether that window's result is
    -- routed to 'm_max' or 'm_avgsum'. Any other value is treated as the
    -- 'OPCODE_POOL_MAX' path (upstream routing guarantees this module only
    -- ever receives pooling opcodes).
    cfg_opcode : in std_ulogic_vector(7 downto 0);
    -- Pool kernel height/width for the in-flight instruction. Contract:
    -- '1 <= cfg_pool_kernel_h, cfg_pool_kernel_w <= g_max_kernel_size'.
    -- 'cfg_pool_kernel_h * cfg_pool_kernel_w' taps (the low bits of
    -- 's_window_m2s.data') are active; the rest of the window beat is
    -- ignored.
    cfg_pool_kernel_h : in std_ulogic_vector(7 downto 0);
    cfg_pool_kernel_w : in std_ulogic_vector(7 downto 0);
    --# {{}}
    -- One pooling window per beat, from 'cnn_accel_window_gen' (already
    -- opcode-selected upstream by a 'handshake_splitter'). 'data' low
    -- 'cfg_pool_kernel_h * cfg_pool_kernel_w * 8' bits hold that many
    -- signed int8 taps, tap 'i' (row-major, 'i = row * cfg_pool_kernel_w +
    -- col') at bits '8*i + 7 downto 8*i'; the remaining high bits are
    -- don't-care.
    s_window_m2s : in axi_stream_m2s_t;
    s_window_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- 'OPCODE_POOL_MAX' result: int8 max over the window's active taps, on
    -- 'data(7 downto 0)' ('data' high bits are 0). To the final output
    -- 'handshake_mux' (bypasses 'cnn_accel_bias_requant').
    m_max_m2s : out axi_stream_m2s_t;
    m_max_s2m : in axi_stream_s2m_t;
    --# {{}}
    -- 'OPCODE_POOL_AVG' result: int32 (g_accum_width-bit) sum over the
    -- window's active taps, on 'data(g_accum_width - 1 downto 0)' ('data'
    -- high bits are 0). To 'cnn_accel_bias_requant' for the pool-area
    -- divide.
    m_avgsum_m2s : out axi_stream_m2s_t;
    m_avgsum_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_pool;

architecture a of cnn_accel_pool is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_max_taps : positive := g_max_kernel_size * g_max_kernel_size;

  type tap_arr_t is array (0 to c_max_taps - 1) of signed(7 downto 0);

  ------------------------------------------------------------------------
  -- Extracts the fixed-size tap array from the low bits of an
  -- 's_window_m2s.data'-shaped vector, per the tap-packing convention
  -- documented on the 's_window_m2s' port above.
  ------------------------------------------------------------------------

  function extract_taps(data : std_ulogic_vector) return tap_arr_t is
    variable result : tap_arr_t;
  begin
    for i in 0 to c_max_taps - 1 loop
      result(i) := signed(data(8 * i + 7 downto 8 * i));
    end loop;
    return result;
  end function;

  -- Linear (unrolled, single combinational stage) max reduction over the
  -- 'active_count' lowest-indexed taps; lanes at/above 'active_count' are
  -- excluded via the loop guard, not via a masked identity value. Not a
  -- literal balanced binary tree (functionally equivalent for a
  -- generic-parameterized lane count) -- see proposal doc section 4.
  function reduce_max(taps : tap_arr_t; active_count : natural) return signed is
    variable result : signed(7 downto 0) := to_signed(-128, 8);
  begin
    for i in 0 to c_max_taps - 1 loop
      if i < active_count and taps(i) > result then
        result := taps(i);
      end if;
    end loop;
    return result;
  end function;

  -- Linear (unrolled) full-precision sum over the 'active_count'
  -- lowest-indexed taps, each sign-extended to 'g_accum_width' before
  -- accumulation -- no intermediate rounding/saturation.
  function reduce_sum(taps : tap_arr_t; active_count : natural) return signed is
    variable result : signed(g_accum_width - 1 downto 0) := (others => '0');
  begin
    for i in 0 to c_max_taps - 1 loop
      if i < active_count then
        result := result + resize(taps(i), g_accum_width);
      end if;
    end loop;
    return result;
  end function;

  ------------------------------------------------------------------------
  -- Combinational reduction over the current 's_window_m2s' beat.
  ------------------------------------------------------------------------

  signal taps : tap_arr_t;
  signal active_count : natural range 0 to c_max_taps;
  signal max_result : signed(7 downto 0);
  signal avgsum_result : signed(g_accum_width - 1 downto 0);
  signal is_avg_sel : std_ulogic;

  ------------------------------------------------------------------------
  -- Single one-entry output register, shared by both output streams:
  -- 'out_is_avg_q' tags which port the held result belongs to. See
  -- proposal doc section 4 for why one tagged register (not two
  -- independent per-port registers) structurally guarantees the
  -- mutual-exclusion requirement.
  ------------------------------------------------------------------------

  signal out_valid_q : std_ulogic := '0';
  signal out_is_avg_q : std_ulogic := '0';
  signal out_last_q : std_ulogic := '0';
  signal out_max_q : signed(7 downto 0) := (others => '0');
  signal out_avgsum_q : signed(g_accum_width - 1 downto 0) := (others => '0');

  -- 'ready' of whichever output port the held (or about-to-be-held)
  -- result targets; gates both the register's own advance and
  -- 's_window_s2m.ready'.
  signal selected_output_ready : std_ulogic;
  signal accepted : std_ulogic;

begin

  ------------------------------------------------------------------------
  -- Static sizing contract checks -- see the generics' doc comments.
  ------------------------------------------------------------------------

  assert c_max_taps * 8 <= axi_stream_data_sz
    report "cnn_accel_pool: g_max_kernel_size**2 * 8 exceeds axi_stream_data_sz " &
      "(the fixed-width axi_stream_pkg data field cannot carry that many taps)"
    severity failure;

  assert g_accum_width <= axi_stream_data_sz
    report "cnn_accel_pool: g_accum_width exceeds axi_stream_data_sz " &
      "(the fixed-width axi_stream_pkg data field cannot carry that wide a sum)"
    severity failure;

  ------------------------------------------------------------------------
  -- Reduction (combinational).
  ------------------------------------------------------------------------

  taps <= extract_taps(s_window_m2s.data);
  active_count <= to_integer(unsigned(cfg_pool_kernel_h)) * to_integer(unsigned(cfg_pool_kernel_w));
  max_result <= reduce_max(taps, active_count);
  avgsum_result <= reduce_sum(taps, active_count);
  is_avg_sel <= '1' when cfg_opcode = OPCODE_POOL_AVG else '0';

  ------------------------------------------------------------------------
  -- One-entry elastic output register: 'accepted' pops 's_window' into the
  -- register whenever it is empty, or is being drained by the selected
  -- output's 'ready' this same cycle (full throughput, no bubble, no loss/
  -- duplication -- see proposal doc section 4/6 for the AXI4-Stream
  -- compliance argument).
  ------------------------------------------------------------------------

  selected_output_ready <= m_avgsum_s2m.ready when out_is_avg_q = '1' else m_max_s2m.ready;
  s_window_s2m.ready <= (not out_valid_q) or selected_output_ready;
  accepted <= s_window_m2s.valid and s_window_s2m.ready;

  register_stage : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        out_valid_q <= '0';
      else
        if out_valid_q = '1' and selected_output_ready = '1' then
          out_valid_q <= '0';
        end if;

        if accepted = '1' then
          out_valid_q <= '1';
          out_is_avg_q <= is_avg_sel;
          out_last_q <= s_window_m2s.last;
          out_max_q <= max_result;
          out_avgsum_q <= avgsum_result;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Output routing: mutually exclusive by construction (one shared
  -- register, one tag bit) -- never both 'valid' in the same cycle.
  ------------------------------------------------------------------------

  m_max_m2s.valid <= out_valid_q and not out_is_avg_q;
  m_max_m2s.last <= out_last_q;
  m_max_m2s.data <= std_ulogic_vector(to_unsigned(0, axi_stream_data_sz - 8)) & std_ulogic_vector(out_max_q);
  m_max_m2s.user <= (others => '-');

  m_avgsum_m2s.valid <= out_valid_q and out_is_avg_q;
  m_avgsum_m2s.last <= out_last_q;
  m_avgsum_m2s.data <=
    std_ulogic_vector(to_unsigned(0, axi_stream_data_sz - g_accum_width)) & std_ulogic_vector(out_avgsum_q);
  m_avgsum_m2s.user <= (others => '-');

end architecture a;
