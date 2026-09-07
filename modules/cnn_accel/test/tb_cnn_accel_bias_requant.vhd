library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- VUnit-5 testbench for cnn_accel_bias_requant. See
-- modules/cnn_accel/doc/cnn_accel_bias_requant_req.md and
-- modules/cnn_accel/doc/cnn_accel_bias_requant_proposal.md sections 8-9
-- for the corner-case list and verification plan. Hand-rolled record-port
-- stimulus/monitor procedures directly against the accum_m2s_t/s2m_t (M6
-- record retrofit) and axi_stream_m2s_t/s2m_t record signals (matching
-- tb_cnn_accel_pool.vhd's/tb_cnn_accel_weight_buffer.vhd's established
-- precedent in this IP, avoiding VUnit's raw axi_stream_master/slave
-- verification components' std_logic-typed flat ports, which would need
-- an extra bridging layer against this module's std_ulogic record ports
-- for no behavioral benefit).
--
-- Expected values are computed by a testbench-local reference function
-- (ref_bias_requantize_relu below), independently transliterated from
-- cnn_accel_model.py's round_shift_right_signed/saturate_signed/
-- bias_requantize_relu -- using numeric_std's 'mod'/'/' operators to get
-- the floor-division quotient/remainder (mirroring Python's divmod
-- directly), rather than the RTL's own shift-and-subtract remainder
-- derivation. This is a structurally different derivation, so agreement
-- between RTL and testbench is a real cross-check, not a tautology.
entity tb_cnn_accel_bias_requant is
  generic (
    -- Independent per-link randomized-backpressure generics, swept per
    -- test in module_cnn_accel.py's setup_vunit (0/0 for the dedicated
    -- full-throughput test, nonzero otherwise): one input, one output
    -- here, so a simple in/out pair rather than tb_cnn_accel_pool.vhd's
    -- three-link split.
    stall_probability_percent_in : natural := 20;
    stall_probability_percent_out : natural := 20;
    -- Output-channel parallelism (lane count). Swept per test in
    -- module_cnn_accel.py's setup_vunit method -- default is a directed
    -- small value (4, not a typical round width together with
    -- c_accum_width=20, so the module cannot silently assume 32/8
    -- anywhere); the M6 record retrofit (accum_m2s_t/s2m_t replacing the
    -- fixed 128-bit axi_stream_m2s_t on 's_accum') removed the old
    -- g_accum_width*g_pe_rows<=128 ceiling, so full_throughput/
    -- backpressure are additionally run at g_pe_rows=8 -- the width the
    -- rest of the design actually uses.
    g_pe_rows : positive := 4;
    runner_cfg : string
  );
end entity tb_cnn_accel_bias_requant;

architecture tb of tb_cnn_accel_bias_requant is

  constant c_accum_width : positive := 20;
  constant c_pe_rows : positive := g_pe_rows;
  constant c_bias_addr_width : positive := 1;
  constant c_max_requant_shift : natural := 31;

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal cfg_bias_en : std_ulogic := '0';
  signal cfg_requant_en : std_ulogic := '0';
  signal cfg_relu_en : std_ulogic := '0';
  signal cfg_requant_scale : std_ulogic_vector(31 downto 0) := (others => '0');
  signal cfg_requant_shift : std_ulogic_vector(7 downto 0) := (others => '0');

  signal bias_rd_addr : std_ulogic_vector(c_bias_addr_width - 1 downto 0);
  signal bias_rd_data : std_ulogic_vector(c_accum_width * c_pe_rows - 1 downto 0) := (others => '0');

  signal s_accum_m2s : accum_m2s_t(data(0 to c_pe_rows - 1)(c_accum_width - 1 downto 0)) :=
    (valid => '0', last => '0', data => (others => (others => '0')));
  signal s_accum_s2m : accum_s2m_t;

  signal m_out_m2s : axi_stream_m2s_t;
  signal m_out_s2m : axi_stream_s2m_t := (ready => '0');

  -- Live stall-probability signals: default to the entity generics, but
  -- overridden locally (and restored afterwards) by test_backpressure so
  -- that test forces real backpressure regardless of whatever the
  -- generics' own default happens to be -- per the proposal doc's
  -- verification-plan promise ("forces a non-zero stall locally
  -- regardless of the generic's default, so a standalone vunit-mcp run
  -- without module_cnn_accel.py's future per-test generic override still
  -- exercises real backpressure").
  signal stall_pct_in : natural := stall_probability_percent_in;
  signal stall_pct_out : natural := stall_probability_percent_out;

  -- Non-blocking scoreboard: one queue for the single output port. The
  -- main process pushes each accepted beat's expected (packed data, last)
  -- pair; monitor_out pops and checks on every accepted output beat.
  constant expected_q : queue_t := new_queue;

  type accum_arr_t is array (0 to c_pe_rows - 1) of
    integer range -(2 ** (c_accum_width - 1)) to (2 ** (c_accum_width - 1) - 1);
  type result_arr_t is array (0 to c_pe_rows - 1) of signed(7 downto 0);

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  ------------------------------------------------------------------------
  -- Independently re-derived golden model. Transliterated from
  -- cnn_accel_model.py's round_shift_right_signed/saturate_signed/
  -- bias_requantize_relu (not copied from this module's own RTL
  -- structure): uses numeric_std's 'mod' (sign-follows-divisor, matching
  -- Python's divmod remainder for a positive divisor) to get the exact
  -- floor-division remainder directly, then an exact '/' for the
  -- quotient (exact because the remainder has already been subtracted
  -- out, so truncating division and floor division agree) -- a
  -- structurally different derivation from the RTL's own shift-and-
  -- subtract-then-compare-twice-remainder approach.
  --
  -- Rounding rule: round-half-up (ties towards +infinity, HW milestone
  -- H0, matching TOSA apply_scale_32 SINGLE_ROUND): +1 iff
  -- 2*remainder >= divisor.
  --
  -- Manually cross-checked during authoring (also cross-checked against
  -- cnn_accel_model.py by hand): accum=1, bias=0, bias_en=0,
  -- requant_scale=16384 (Q15 0.5), requant_shift=0 (combined shift 15):
  -- product=16384; 16384 mod 32768 = 16384 (an exact tie,
  -- twice_remainder=32768=divisor); quotient=0 -> rounds up to 1.
  -- accum=-1: product=-16384; -16384 mod 32768 = 16384 (tie), floor
  -- quotient=-1 -> rounds up to 0. Both this function and
  -- cnn_accel_model.py's round_shift_right_signed agree.
  ------------------------------------------------------------------------

  function ref_round_shift_right(value : signed; shift : natural) return signed is
    constant ext_width : positive := value'length + 1;
    variable value_ext : signed(ext_width - 1 downto 0);
    variable divisor : signed(ext_width - 1 downto 0);
    variable remainder : signed(ext_width - 1 downto 0);
    variable quotient : signed(ext_width - 1 downto 0);
    variable remainder_wide : signed(ext_width downto 0);
    variable twice_remainder : signed(ext_width downto 0);
    variable divisor_wide : signed(ext_width downto 0);
  begin
    assert shift >= 1
      report "ref_round_shift_right: this module's combined shift is always >= 15, shift<=0 is not exercised"
      severity failure;

    value_ext := resize(value, ext_width);
    divisor := shift_left(to_signed(1, ext_width), shift);

    -- numeric_std 'mod': result's sign follows the right operand
    -- (divisor, always positive here), so remainder is in [0, divisor) --
    -- exactly Python's divmod(value, divisor) remainder convention.
    remainder := value_ext mod divisor;
    -- Exact division: (value_ext - remainder) is exactly divisible by
    -- divisor by construction, so truncating '/' and floor division
    -- agree here regardless of sign.
    quotient := (value_ext - remainder) / divisor;

    remainder_wide := resize(remainder, ext_width + 1);
    twice_remainder := shift_left(remainder_wide, 1);
    divisor_wide := resize(divisor, ext_width + 1);

    if twice_remainder < divisor_wide then
      return resize(quotient, value'length);
    else
      -- >= half, exact ties included: round towards +infinity.
      return resize(quotient + 1, value'length);
    end if;
  end function;

  function ref_saturate_signed(value : signed; result_width : positive) return signed is
    variable min_value_ext : signed(value'range) := to_signed(-(2 ** (result_width - 1)), value'length);
    variable max_value_ext : signed(value'range) := to_signed(2 ** (result_width - 1) - 1, value'length);
    variable clamped : signed(value'range);
  begin
    if value < min_value_ext then
      clamped := min_value_ext;
    elsif value > max_value_ext then
      clamped := max_value_ext;
    else
      clamped := value;
    end if;
    return resize(clamped, result_width);
  end function;

  -- Per-lane reference function for cnn_accel_bias_requant. See
  -- cnn_accel_model.py's bias_requantize_relu docstring: int32 accumulator
  -- -> (+ bias) -> (x requant_scale, Q15) -> (>> requant_shift, rounded)
  -- -> saturate to int8 -> (optional ReLU clamp at 0, before the int8
  -- saturate). requant_en='0' bypasses scaling: bias/ReLU still applied,
  -- then the result is SATURATED to int8 (same overflow semantic as the
  -- requant_en='1' path -- architectural decision D2, single overflow
  -- semantic across both paths, matching the golden model's fix #1).
  function ref_bias_requantize_relu(
    accum : signed;
    bias : signed;
    bias_en : std_ulogic;
    requant_en : std_ulogic;
    relu_en : std_ulogic;
    requant_scale : signed(31 downto 0);
    requant_shift : natural
  ) return signed is
    constant sum_width : positive := accum'length + 1;
    constant product_width : positive := sum_width + requant_scale'length;
    variable total : signed(sum_width - 1 downto 0);
    variable product : signed(product_width - 1 downto 0);
    variable scaled : signed(product_width - 1 downto 0);
    variable bypass_total : signed(sum_width - 1 downto 0);
  begin
    if bias_en = '1' then
      total := resize(accum, sum_width) + resize(bias, sum_width);
    else
      total := resize(accum, sum_width);
    end if;

    if requant_en = '1' then
      product := total * requant_scale;
      scaled := ref_round_shift_right(product, 15 + requant_shift);

      if relu_en = '1' and scaled(scaled'high) = '1' then
        scaled := (others => '0');
      end if;

      return ref_saturate_signed(scaled, 8);
    else
      bypass_total := total;

      if relu_en = '1' and bypass_total(bypass_total'high) = '1' then
        bypass_total := (others => '0');
      end if;

      return ref_saturate_signed(bypass_total, 8);
    end if;
  end function;

  ------------------------------------------------------------------------
  -- Packing helpers. 's_accum_m2s.data' is now (M6 record retrofit) an
  -- array of lanes (accum_array_t), so it is built directly, one
  -- element per lane, rather than packed into a flat vector -- 'bias_rd_data'
  -- (a plain std_ulogic_vector port, unaffected by this retrofit) and the
  -- int8 output ('m_out_m2s.data') are still genuinely flat AXI4-Stream
  -- payloads, so those two keep the old bit-packed convention: lane 'l'
  -- occupies bits [w*(l+1)-1 downto w*l], ascending from the low bits,
  -- zero-padded above the active lanes.
  ------------------------------------------------------------------------

  function to_accum_array(values : accum_arr_t) return accum_array_t is
    variable result : accum_array_t(0 to c_pe_rows - 1)(c_accum_width - 1 downto 0);
  begin
    for i in 0 to c_pe_rows - 1 loop
      result(i) := to_signed(values(i), c_accum_width);
    end loop;
    return result;
  end function;

  -- Sized exactly to 'bias_rd_data's actual port width
  -- (g_accum_width*g_pe_rows, no axi_stream_data_sz padding -- that port
  -- is not an AXI4-Stream 'data' field).
  function pack_lanes_bias(values : accum_arr_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(c_accum_width * c_pe_rows - 1 downto 0);
  begin
    for i in 0 to c_pe_rows - 1 loop
      result(c_accum_width * (i + 1) - 1 downto c_accum_width * i) :=
        std_ulogic_vector(to_signed(values(i), c_accum_width));
    end loop;
    return result;
  end function;

  function pack_lanes_result(values : result_arr_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(axi_stream_data_sz - 1 downto 0) := (others => '0');
  begin
    for i in 0 to c_pe_rows - 1 loop
      result(8 * (i + 1) - 1 downto 8 * i) := std_ulogic_vector(values(i));
    end loop;
    return result;
  end function;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  -- Structural safety net: 'bias_rd_addr' must stay constant all-zeros
  -- for the whole simulation (v1 design decision, see proposal doc
  -- section 3), not just incidentally true during the directed tests --
  -- redundant with (not a substitute for) the functional scoreboard.
  ------------------------------------------------------------------------

  bias_rd_addr_check : process(clk)
  begin
    if rising_edge(clk) and reset = '0' then
      check_equal(
        bias_rd_addr, std_ulogic_vector'(bias_rd_addr'range => '0'),
        "bias_rd_addr must stay constant all-zeros (v1 single-bias-row design)"
      );
    end if;
  end process;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_bias_requant
    generic map (
      g_accum_width => c_accum_width,
      g_pe_rows => c_pe_rows,
      g_bias_addr_width => c_bias_addr_width,
      g_max_requant_shift => c_max_requant_shift
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_bias_en => cfg_bias_en,
      cfg_requant_en => cfg_requant_en,
      cfg_relu_en => cfg_relu_en,
      cfg_requant_scale => cfg_requant_scale,
      cfg_requant_shift => cfg_requant_shift,

      bias_rd_addr => bias_rd_addr,
      bias_rd_data => bias_rd_data,

      s_accum_m2s => s_accum_m2s,
      s_accum_s2m => s_accum_s2m,

      m_out_m2s => m_out_m2s,
      m_out_s2m => m_out_s2m
    );

  ------------------------------------------------------------------------
  -- Output monitor: independent randomized-'ready' responder. Pops and
  -- checks the scoreboard queue on every accepted output beat.
  ------------------------------------------------------------------------

  monitor_out : process
    variable rnd : RandomPType;
    variable expected_data : std_ulogic_vector(axi_stream_data_sz - 1 downto 0);
    variable expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_monitor");
    m_out_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < stall_pct_out then
        m_out_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_out_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_out_m2s.valid = '1' and m_out_s2m.ready = '1' then
        expected_data := pop(expected_q);
        expected_last := pop(expected_q);
        check_equal(m_out_m2s.data, expected_data, "m_out data mismatch");
        check_equal(m_out_m2s.last, expected_last, "m_out last mismatch");
      end if;
    end loop;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
    end procedure;

    procedure random_accum_arr(p_arr : out accum_arr_t) is
    begin
      for i in 0 to c_pe_rows - 1 loop
        p_arr(i) := rnd.RandInt(-(2 ** (c_accum_width - 1)), 2 ** (c_accum_width - 1) - 1);
      end loop;
    end procedure;

    -- Pushes one 's_accum' beat (with randomized input-side stall,
    -- driven by 'stall_pct_in') and enqueues its expected per-lane result
    -- (computed by ref_bias_requantize_relu, not the RTL) onto the
    -- scoreboard queue.
    procedure send_beat(
      accum_vals : accum_arr_t;
      bias_vals : accum_arr_t;
      bias_en : std_ulogic;
      requant_en : std_ulogic;
      relu_en : std_ulogic;
      scale : signed(31 downto 0);
      shift : natural;
      beat_last : std_ulogic
    ) is
      variable results : result_arr_t;
    begin
      cfg_bias_en <= bias_en;
      cfg_requant_en <= requant_en;
      cfg_relu_en <= relu_en;
      cfg_requant_scale <= std_ulogic_vector(scale);
      cfg_requant_shift <= std_ulogic_vector(to_unsigned(shift, 8));

      s_accum_m2s.data <= to_accum_array(accum_vals);
      bias_rd_data <= pack_lanes_bias(bias_vals);
      s_accum_m2s.last <= beat_last;

      if rnd.RandInt(0, 99) < stall_pct_in then
        s_accum_m2s.valid <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      s_accum_m2s.valid <= '1';
      wait until rising_edge(clk) and s_accum_s2m.ready = '1';
      s_accum_m2s.valid <= '0';

      for i in 0 to c_pe_rows - 1 loop
        results(i) := ref_bias_requantize_relu(
          to_signed(accum_vals(i), c_accum_width),
          to_signed(bias_vals(i), c_accum_width),
          bias_en, requant_en, relu_en,
          scale, shift
        );
      end loop;

      push(expected_q, pack_lanes_result(results));
      push(expected_q, beat_last);
    end procedure;

    -- Convenience: same directed value on every lane (mirrors
    -- tb_cnn_accel_pool.vhd's run_directed_extremes idiom).
    procedure send_directed(
      accum_val : integer;
      bias_val : integer;
      bias_en : std_ulogic;
      requant_en : std_ulogic;
      relu_en : std_ulogic;
      scale : signed(31 downto 0);
      shift : natural
    ) is
      variable accum_vals, bias_vals : accum_arr_t;
    begin
      for i in 0 to c_pe_rows - 1 loop
        accum_vals(i) := accum_val;
        bias_vals(i) := bias_val;
      end loop;
      send_beat(accum_vals, bias_vals, bias_en, requant_en, relu_en, scale, shift, '0');
    end procedure;

    -- Waits (bounded) until the scoreboard queue has drained, then
    -- confirms it is truly empty (every expected output actually
    -- arrived) rather than just timing out.
    procedure drain_and_check(max_wait_cycles : positive) is
      variable cycles : natural := 0;
    begin
      while not is_empty(expected_q) and cycles < max_wait_cycles loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
      end loop;
      check_true(is_empty(expected_q), "m_out scoreboard queue did not drain in time");
    end procedure;

    constant c_scale_half : signed(31 downto 0) := to_signed(16384, 32); -- Q15 0.5
    constant c_scale_one : signed(31 downto 0) := to_signed(32768, 32); -- Q15 1.0

    variable accum_vals, bias_vals : accum_arr_t;
    variable scale : signed(31 downto 0);
    variable shift : natural;
    variable start_time : time;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    do_reset;
    wait until rising_edge(clk);

    if run("test_round_half_up_ties") then
      -- Directed ties (both quotient parities, both signs) at combined
      -- shift=15 (requant_shift=0), scale=0.5 Q15: product=accum*16384,
      -- remainder exactly half the divisor (32768) whenever accum is odd.
      -- Half-up: every tie rounds towards +infinity (H0).
      -- accum=1 -> product=16384 (0.5), quotient=0 -> 1.
      send_directed(1, 0, '0', '1', '0', c_scale_half, 0);
      -- accum=3 -> product=49152 (1.5), quotient=1 -> 2.
      send_directed(3, 0, '0', '1', '0', c_scale_half, 0);
      -- accum=-1 -> product=-16384 (-0.5), floor quotient=-1 -> 0.
      send_directed(-1, 0, '0', '1', '0', c_scale_half, 0);
      -- accum=-3 -> product=-49152 (-1.5), floor quotient=-2 -> -1.
      send_directed(-3, 0, '0', '1', '0', c_scale_half, 0);
      drain_and_check(200);

    elsif run("test_saturate_both_directions") then
      -- Directed boundary values at scale=1.0, shift=0 (identity requant):
      -- 127/-128 are exactly representable (no saturation); 128/-129
      -- (and further out) must clamp.
      send_directed(127, 0, '0', '1', '0', c_scale_one, 0);
      send_directed(128, 0, '0', '1', '0', c_scale_one, 0);
      send_directed(-128, 0, '0', '1', '0', c_scale_one, 0);
      send_directed(-129, 0, '0', '1', '0', c_scale_one, 0);
      send_directed(200, 0, '0', '1', '0', c_scale_one, 0);
      send_directed(-200, 0, '0', '1', '0', c_scale_one, 0);

      -- Randomized saturation stress: full accum range with a
      -- deliberately huge scale (2**20), shift=0, so almost every beat
      -- lands well outside the int8 range in one direction or the other.
      for beat in 0 to 39 loop
        random_accum_arr(accum_vals);
        for i in 0 to c_pe_rows - 1 loop
          bias_vals(i) := 0;
        end loop;
        send_beat(accum_vals, bias_vals, '0', '1', '0', to_signed(2 ** 20, 32), 0, '0');
      end loop;
      drain_and_check(200);

    elsif run("test_relu_before_saturate") then
      -- Large positive: ReLU must not affect it, and it still saturates
      -- at +127 (not some ReLU-specific unbounded upper range).
      send_directed(1000, 0, '0', '1', '1', c_scale_one, 0);
      -- Large negative: ReLU clamps to 0 (not -128, which is what it
      -- would saturate to without ReLU).
      send_directed(-1000, 0, '0', '1', '1', c_scale_one, 0);
      -- Moderate positive, no saturation either way: passes through
      -- unaffected by ReLU.
      send_directed(50, 0, '0', '1', '1', c_scale_one, 0);
      -- Moderate negative: ReLU clamps to 0.
      send_directed(-50, 0, '0', '1', '1', c_scale_one, 0);
      -- Same magnitude negative without ReLU, for contrast: must NOT be
      -- clamped to 0 (saturate_signed only, unaffected by ReLU when
      -- relu_en='0').
      send_directed(-50, 0, '0', '1', '0', c_scale_one, 0);
      drain_and_check(200);

    elsif run("test_flag_combinations") then
      -- All 8 combinations of cfg_bias_en/cfg_requant_en/cfg_relu_en,
      -- randomized accum/bias/scale/shift per beat within each combo.
      for combo in 0 to 7 loop
        for beat in 0 to 14 loop
          random_accum_arr(accum_vals);
          random_accum_arr(bias_vals);
          scale := rnd.RandSigned(32);
          shift := rnd.RandInt(0, c_max_requant_shift);
          send_beat(
            accum_vals, bias_vals,
            to_sl((combo mod 2) = 1),
            to_sl(((combo / 2) mod 2) = 1),
            to_sl(((combo / 4) mod 2) = 1),
            scale, shift, '0'
          );
        end loop;
      end loop;
      drain_and_check(500);

    elsif run("test_bypass_saturates") then
      -- Directed: cfg_requant_en='0' must SATURATE to int8, not wrap
      -- (architectural decision D2 -- a single overflow semantic shared
      -- with the requant_en='1' path, matching the golden model's fix
      -- #1). Exact boundary values +127/-128 pass through unchanged;
      -- anything beyond clamps.
      send_directed(127, 0, '0', '0', '0', c_scale_one, 0);
      send_directed(128, 0, '0', '0', '0', c_scale_one, 0);
      send_directed(-128, 0, '0', '0', '0', c_scale_one, 0);
      send_directed(-129, 0, '0', '0', '0', c_scale_one, 0);
      send_directed(300, 0, '0', '0', '0', c_scale_one, 0);
      send_directed(-300, 0, '0', '0', '0', c_scale_one, 0);
      -- With bias added and ReLU applied to the bypass path.
      send_directed(200, 50, '1', '0', '1', c_scale_one, 0);
      send_directed(-200, -50, '1', '0', '1', c_scale_one, 0);

      -- Randomized batch, bias/ReLU toggling, requant_en fixed at '0'.
      for beat in 0 to 39 loop
        random_accum_arr(accum_vals);
        random_accum_arr(bias_vals);
        send_beat(
          accum_vals, bias_vals,
          to_sl(rnd.RandInt(0, 1) = 1), '0', to_sl(rnd.RandInt(0, 1) = 1),
          c_scale_one, 0, '0'
        );
      end loop;
      drain_and_check(200);

    elsif run("test_bypass_relu_before_saturate") then
      -- Mirrors test_relu_before_saturate, but for the cfg_requant_en='0'
      -- bypass path: ReLU must clamp negative totals to 0 BEFORE the int8
      -- saturate, not after (a large negative total would otherwise
      -- saturate to -128).
      send_directed(1000, 0, '0', '0', '1', c_scale_one, 0);
      -- Large negative: ReLU clamps to 0 (not -128, which is what it
      -- would saturate to without ReLU).
      send_directed(-1000, 0, '0', '0', '1', c_scale_one, 0);
      -- Moderate positive, no saturation either way: passes through
      -- unaffected by ReLU.
      send_directed(50, 0, '0', '0', '1', c_scale_one, 0);
      -- Moderate negative: ReLU clamps to 0.
      send_directed(-50, 0, '0', '0', '1', c_scale_one, 0);
      -- Same magnitude negative without ReLU, for contrast: must NOT be
      -- clamped to 0 (passes through as -50, unaffected by relu_en='0').
      send_directed(-50, 0, '0', '0', '0', c_scale_one, 0);
      drain_and_check(200);

    elsif run("test_full_throughput") then
      -- Zero stall on both links (generic-driven, per
      -- module_cnn_accel.py's future per-test config): must sustain one
      -- output beat per accepted input beat.
      start_time := now;
      for beat in 0 to 299 loop
        random_accum_arr(accum_vals);
        random_accum_arr(bias_vals);
        scale := rnd.RandSigned(32);
        shift := rnd.RandInt(0, c_max_requant_shift);
        send_beat(
          accum_vals, bias_vals,
          to_sl(rnd.RandInt(0, 1) = 1), to_sl(rnd.RandInt(0, 1) = 1), to_sl(rnd.RandInt(0, 1) = 1),
          scale, shift, to_sl(beat = 299)
        );
      end loop;
      drain_and_check(350);

      check_relation(
        (now - start_time) < 320 * c_clk_period,
        "cnn_accel_bias_requant did not sustain full throughput at zero stall"
      );

    elsif run("test_backpressure") then
      -- Forces a non-zero stall locally on both links regardless of the
      -- generics' own default (see the signal declarations' comment),
      -- so this test exercises real backpressure even in a standalone
      -- vunit-mcp run without module_cnn_accel.py's future per-test
      -- generic override.
      stall_pct_in <= 40;
      stall_pct_out <= 40;
      wait until rising_edge(clk);

      for beat in 0 to 149 loop
        random_accum_arr(accum_vals);
        random_accum_arr(bias_vals);
        scale := rnd.RandSigned(32);
        shift := rnd.RandInt(0, c_max_requant_shift);
        send_beat(
          accum_vals, bias_vals,
          to_sl(rnd.RandInt(0, 1) = 1), to_sl(rnd.RandInt(0, 1) = 1), to_sl(rnd.RandInt(0, 1) = 1),
          scale, shift, to_sl(beat = 149)
        );
      end loop;
      drain_and_check(2000);

      stall_pct_in <= stall_probability_percent_in;
      stall_pct_out <= stall_probability_percent_out;

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
