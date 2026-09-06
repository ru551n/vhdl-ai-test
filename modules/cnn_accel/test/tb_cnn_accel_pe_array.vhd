library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- VUnit-5 testbench for cnn_accel_pe_array ("M4", the tiled-dataflow
-- rewrite). See modules/cnn_accel/doc/cnn_accel_pe_array_req.md,
-- doc/cnn_accel_pe_array_proposal.md and
-- doc/cnn_accel_tiled_dataflow_proposal.md sections 3/4/7 (the RATIFIED
-- tiled design this DUT implements).
--
-- Hand-rolled record-port stimulus/monitor procedures directly against
-- 'window_m2s_t'/'window_s2m_t' and 'accum_m2s_t'/'accum_s2m_t' (no VUnit
-- axi_stream_master/slave VC -- matches tb_cnn_accel_window_gen.vhd's/
-- tb_cnn_accel_bias_requant.vhd's identical choice for this IP's own
-- record-typed links). The 'weight_rd_addr'/'weight_rd_data' port pair is
-- not a handshaked link at all (cnn_accel_weight_buffer.vhd's own
-- contract: unconditional, always-ready, 1-cycle registered read), so
-- this testbench models it directly with a small synchronous "memory"
-- process rather than instantiating the real cnn_accel_weight_buffer --
-- exercising exactly the port contract this DUT actually depends on.
--
-- Golden model: an independent, purely-integer per-beat/per-pixel
-- reference (golden_beat_partial below) that mirrors the *specified*
-- weight-row addressing contract (first_tile resets the row base to 0,
-- every beat's own num_groups advances it -- doc/cnn_accel_
-- tiled_dataflow_proposal.md section 4) and the *specified* broadcast-
-- activation/per-lane-weight MAC grouping (proposal doc section 6), but
-- using plain host-language integers rather than the RTL's own signed/
-- resize arithmetic -- a structurally independent re-derivation, not a
-- transliteration of cnn_accel_pe_array.vhd's own compute_partial_sums().
--
-- Generics deliberately deviate from module_cnn_accel.py's reference
-- hardware point (g_pe_rows=g_pe_cols=8, g_tile_channels=8): c_tile_channels
-- (6) is NOT a multiple of c_pe_cols (4), so every kernel shape with an odd
-- kh*kw exercises a beat whose 'mac_taps' does not divide evenly by
-- 'g_pe_cols' -- the exact masking corner case flagged in
-- cnn_accel_pe_array.vhd's own compute_partial_sums() comment.
entity tb_cnn_accel_pe_array is
  generic (
    -- Independent per-link randomized-backpressure generics, swept per
    -- test in module_cnn_accel.py's setup_vunit (0/0 for the dedicated
    -- full-throughput test, nonzero otherwise) -- mirrors
    -- tb_cnn_accel_bias_requant.vhd's identical in/out generic pair
    -- ('s_window' is the one input link, 'm_accum' the one output link;
    -- 'weight_rd_addr'/'weight_rd_data' is not a handshaked link, so has
    -- no stall generic of its own).
    stall_probability_percent_in : natural := 20;
    stall_probability_percent_out : natural := 20;
    runner_cfg : string
  );
end entity tb_cnn_accel_pe_array;

architecture tb of tb_cnn_accel_pe_array is

  -- Small, directed generics -- see this file's header comment for why
  -- c_tile_channels is deliberately not a multiple of c_pe_cols.
  constant c_pe_rows : positive := 4;
  constant c_pe_cols : positive := 4;
  constant c_kernel_max : positive := 3;
  constant c_tile_channels : positive := 6;
  constant c_accum_width : positive := 32;
  -- Sized so even the largest kernel exercised here (3x3, mac_taps=54,
  -- num_groups=14) times the largest T tested (4) fits with headroom
  -- (14*4=56 <= 128) -- see this file's run_pixel/test bodies.
  constant c_weight_buffer_depth : positive := 128;

  constant c_max_window_len : positive := window_data_length(c_kernel_max, c_tile_channels);
  constant c_addr_width : positive := num_bits_needed(c_weight_buffer_depth - 1);
  constant c_weight_lanes : positive := c_pe_rows * c_pe_cols;

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal cfg_kernel_h : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_kernel_w : std_ulogic_vector(7 downto 0) := (others => '0');

  signal s_window_m2s : window_m2s_t(data(0 to c_max_window_len - 1)) := (
    valid => '0', last => '0', first_tile => '0', last_tile => '0',
    data => (others => (others => '0'))
  );
  signal s_window_s2m : window_s2m_t;

  signal weight_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0);
  signal weight_rd_data : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0) := (others => '0');

  signal m_accum_m2s : accum_m2s_t(data(0 to c_pe_rows - 1)(c_accum_width - 1 downto 0));
  signal m_accum_s2m : accum_s2m_t := (ready => '0');

  -- Live stall-probability signals: default to the entity generics, but
  -- overridden locally (and restored afterwards) by test_backpressure --
  -- mirrors tb_cnn_accel_bias_requant.vhd's identical idiom.
  signal stall_pct_in : natural := stall_probability_percent_in;
  signal stall_pct_out : natural := stall_probability_percent_out;

  -- Non-blocking scoreboard: one queue for the single 'm_accum' output
  -- link. run_pixel pushes each pixel's expected (packed accum data, last)
  -- pair once its final tile beat is issued; monitor_out pops and checks
  -- on every accepted output beat.
  constant expected_q : queue_t := new_queue;

  -- Weight "memory" content, modelling exactly cnn_accel_weight_buffer's
  -- own read port contract (registered, 1-cycle latency) without
  -- instantiating that whole module -- see this file's header comment.
  -- Lane 'l = r*c_pe_cols + c' (row-major), matching cnn_accel_pe_array's
  -- own 'weight_rd_data' lane layout.
  type weight_row_t is array (0 to c_weight_lanes - 1) of integer range -128 to 127;
  type weight_mem_t is array (0 to c_weight_buffer_depth - 1) of weight_row_t;
  signal weight_mem_s : weight_mem_t := (others => (others => 0));

  type window_row_t is array (0 to c_max_window_len - 1) of integer range -128 to 127;
  type accum_int_arr_t is array (0 to c_pe_rows - 1) of integer;

  -- Kernel shapes swept by test_varying_kernels: 1x1 up to
  -- c_kernel_max x c_kernel_max, square and non-square -- mirrors
  -- tb_cnn_accel_window_gen.vhd's/tb_cnn_accel_pool.vhd's identical
  -- 'c_shapes' idiom.
  type shape_t is record
    h : natural;
    w : natural;
  end record;
  type shape_arr_t is array (natural range <>) of shape_t;
  constant c_shapes : shape_arr_t(0 to 4) := (
    (1, 1), (2, 2), (3, 3), (2, 3), (3, 2)
  );

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  ------------------------------------------------------------------------
  -- Packing helpers.
  ------------------------------------------------------------------------

  -- Packs one weight row (c_weight_lanes int8 lanes) into
  -- 'weight_rd_data's own bit layout (lane l at bits 8*(l+1)-1 downto 8*l).
  function pack_weight_row(row : weight_row_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);
  begin
    for i in 0 to c_weight_lanes - 1 loop
      result(8 * (i + 1) - 1 downto 8 * i) := std_ulogic_vector(to_signed(row(i), 8));
    end loop;
    return result;
  end function;

  -- Packs one pixel's expected per-row accumulator results, same lane
  -- layout ('m_accum_m2s.data' element 'r' at bits
  -- 'c_accum_width*(r+1)-1 downto c_accum_width*r' once flattened).
  function pack_expected(values : accum_int_arr_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(c_accum_width * c_pe_rows - 1 downto 0);
  begin
    for i in 0 to c_pe_rows - 1 loop
      result(c_accum_width * (i + 1) - 1 downto c_accum_width * i) :=
        std_ulogic_vector(to_signed(values(i), c_accum_width));
    end loop;
    return result;
  end function;

  -- Flattens the DUT's own 'm_accum_m2s.data' (an array of possibly-wide
  -- 'signed' elements, cnn_accel_pkg.vhd's 'accum_array_t') into the same
  -- flat layout as pack_expected, for a single check_equal comparison.
  function pack_actual(data : accum_array_t) return std_ulogic_vector is
    constant elem_width : positive := data(data'left)'length;
    variable result : std_ulogic_vector(data'length * elem_width - 1 downto 0);
    variable idx : natural := 0;
  begin
    for i in data'range loop
      result(elem_width * (idx + 1) - 1 downto elem_width * idx) := std_ulogic_vector(data(i));
      idx := idx + 1;
    end loop;
    return result;
  end function;

  ------------------------------------------------------------------------
  -- Golden model: one beat's (one tile's) contribution to each output
  -- row's running sum. 'weight_base' is the row this beat's group
  -- sequence starts at (0 on a pixel's first_tile beat, otherwise the
  -- pixel's running total of every earlier beat's own num_groups -- see
  -- run_pixel below, which tracks this exactly per doc/cnn_accel_
  -- tiled_dataflow_proposal.md section 4's contract). Structurally
  -- independent from cnn_accel_pe_array.vhd's own
  -- compute_partial_sums(): plain integers, no signed/resize/clamped-idx
  -- machinery, looping over the same (group, row, col) shape the
  -- proposal doc's broadcast-activation/per-lane-weight contract
  -- specifies.
  ------------------------------------------------------------------------

  function golden_beat_partial(
    window : window_row_t;
    weight_mem : weight_mem_t;
    weight_base : natural;
    mac_taps : natural
  ) return accum_int_arr_t is
    variable result : accum_int_arr_t := (others => 0);
    variable num_groups : natural := (mac_taps + c_pe_cols - 1) / c_pe_cols;
    variable idx : natural;
    variable row_idx : natural;
  begin
    for g in 0 to num_groups - 1 loop
      row_idx := weight_base + g;
      for r in 0 to c_pe_rows - 1 loop
        for c in 0 to c_pe_cols - 1 loop
          idx := g * c_pe_cols + c;
          if idx < mac_taps then
            result(r) := result(r) + window(idx) * weight_mem(row_idx)(r * c_pe_cols + c);
          end if;
        end loop;
      end loop;
    end loop;
    return result;
  end function;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_pe_array
    generic map (
      g_pe_rows => c_pe_rows,
      g_pe_cols => c_pe_cols,
      g_accum_width => c_accum_width,
      g_max_kernel_size => c_kernel_max,
      g_tile_channels => c_tile_channels,
      g_weight_buffer_depth => c_weight_buffer_depth
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_kernel_h => cfg_kernel_h,
      cfg_kernel_w => cfg_kernel_w,

      s_window_m2s => s_window_m2s,
      s_window_s2m => s_window_s2m,

      weight_rd_addr => weight_rd_addr,
      weight_rd_data => weight_rd_data,

      m_accum_m2s => m_accum_m2s,
      m_accum_s2m => m_accum_s2m
    );

  ------------------------------------------------------------------------
  -- Weight "memory" model: registered, 1-cycle read latency, exactly
  -- cnn_accel_weight_buffer.vhd's own read port contract (see this file's
  -- header comment). Always responds, regardless of 'reset' (matches the
  -- real component: the read port is a plain synchronous read, not part
  -- of any reset-cleared handshake state).
  ------------------------------------------------------------------------

  weight_mem_model : process(clk)
    variable addr_int : natural;
  begin
    if rising_edge(clk) then
      addr_int := to_integer(unsigned(weight_rd_addr));
      if addr_int <= c_weight_buffer_depth - 1 then
        weight_rd_data <= pack_weight_row(weight_mem_s(addr_int));
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Output monitor: independent randomized-'ready' responder. Pops and
  -- checks the scoreboard queue on every accepted 'm_accum' beat.
  ------------------------------------------------------------------------

  monitor_out : process
    variable rnd : RandomPType;
    variable expected_data : std_ulogic_vector(c_accum_width * c_pe_rows - 1 downto 0);
    variable expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_monitor");
    m_accum_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < stall_pct_out then
        m_accum_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_accum_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_accum_m2s.valid = '1' and m_accum_s2m.ready = '1' then
        expected_data := pop(expected_q);
        expected_last := pop(expected_q);
        check_equal(pack_actual(m_accum_m2s.data), expected_data, "m_accum data mismatch");
        check_equal(m_accum_m2s.last, expected_last, "m_accum last mismatch");
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

    -- Randomizes every row of the weight "memory" -- called once, before
    -- any test streams a beat, and never again (a static weight image for
    -- the whole simulation, mirroring a real weight_buffer bank already
    -- filled before compute starts).
    procedure randomize_weight_mem is
    begin
      for row in 0 to c_weight_buffer_depth - 1 loop
        for lane in 0 to c_weight_lanes - 1 loop
          weight_mem_s(row)(lane) <= rnd.RandInt(-128, 127);
        end loop;
      end loop;
    end procedure;

    -- Randomizes a window's first 'mac_taps' taps; taps at/beyond
    -- 'mac_taps' (only possible when 'c_pe_cols' does not divide
    -- 'mac_taps' evenly -- this file's own c_tile_channels/c_pe_cols
    -- mismatch, see header comment) get deliberately *non-zero* garbage,
    -- not 0: the point is to prove the DUT's own masking (not a
    -- coincidentally-zero source) is what keeps those lanes out of the
    -- sum -- mirrors tb_cnn_accel_window_gen.vhd's identical
    -- 'push_tile' idiom.
    procedure random_window(window : out window_row_t; mac_taps : natural) is
    begin
      for i in 0 to c_max_window_len - 1 loop
        if i < mac_taps then
          window(i) := rnd.RandInt(-128, 127);
        else
          window(i) := rnd.RandInt(1, 127);
        end if;
      end loop;
    end procedure;

    -- Pushes one 's_window' beat, with randomized input-side stall
    -- (driven by 'stall_pct_in').
    procedure send_window_beat(
      window : window_row_t;
      kh, kw : natural;
      first_tile, last_tile, beat_last : std_ulogic
    ) is
    begin
      cfg_kernel_h <= std_ulogic_vector(to_unsigned(kh, 8));
      cfg_kernel_w <= std_ulogic_vector(to_unsigned(kw, 8));
      for i in 0 to c_max_window_len - 1 loop
        s_window_m2s.data(i) <= std_ulogic_vector(to_signed(window(i), 8));
      end loop;
      s_window_m2s.first_tile <= first_tile;
      s_window_m2s.last_tile <= last_tile;
      s_window_m2s.last <= beat_last;

      if rnd.RandInt(0, 99) < stall_pct_in then
        s_window_m2s.valid <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      s_window_m2s.valid <= '1';
      wait until rising_edge(clk) and s_window_s2m.ready = '1';
      s_window_m2s.valid <= '0';
    end procedure;

    -- Streams one output pixel needing 'num_tiles' input-channel tiles
    -- (T=1 when 'num_tiles=1', both first_tile/last_tile='1' on its one
    -- beat -- window_m2s_t's own documented contract), same (kh, kw) for
    -- every tile of the pixel (the only realistic use -- a layer's kernel
    -- shape does not change mid-pixel). Tracks the pixel's own running
    -- weight-row base exactly per doc/cnn_accel_tiled_dataflow_
    -- proposal.md section 4 (0 on first_tile, advanced by each beat's own
    -- num_groups afterwards) and its running accumulator sum (golden_
    -- beat_partial), pushing the expected (packed sum, last) pair onto
    -- the scoreboard once the final tile beat is issued.
    procedure run_pixel(kh, kw, num_tiles : natural; is_last_pixel : boolean) is
      constant mac_taps : natural := kh * kw * c_tile_channels;
      constant num_groups : natural := (mac_taps + c_pe_cols - 1) / c_pe_cols;
      variable weight_base : natural := 0;
      variable accum : accum_int_arr_t := (others => 0);
      variable beat_partial : accum_int_arr_t;
      variable window : window_row_t;
      variable beat_last : std_ulogic;
    begin
      for tile in 0 to num_tiles - 1 loop
        random_window(window, mac_taps);
        beat_last := to_sl(is_last_pixel and tile = num_tiles - 1);
        send_window_beat(
          window, kh, kw,
          to_sl(tile = 0), to_sl(tile = num_tiles - 1), beat_last
        );

        beat_partial := golden_beat_partial(window, weight_mem_s, weight_base, mac_taps);
        for r in 0 to c_pe_rows - 1 loop
          accum(r) := accum(r) + beat_partial(r);
        end loop;
        weight_base := weight_base + num_groups;

        if tile = num_tiles - 1 then
          push(expected_q, pack_expected(accum));
          push(expected_q, beat_last);
        end if;
      end loop;
    end procedure;

    -- Waits (bounded) until 'expected_q' has drained, then confirms it is
    -- truly empty (every expected output actually arrived).
    procedure drain_and_check(max_wait_cycles : positive) is
      variable cycles : natural := 0;
    begin
      while not is_empty(expected_q) and cycles < max_wait_cycles loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
      end loop;
      check_true(is_empty(expected_q), "m_accum scoreboard queue did not drain in time");
    end procedure;

    variable start_time : time;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    randomize_weight_mem;
    do_reset;
    wait until rising_edge(clk);

    if run("test_single_tile_basic") then
      -- T=1: both first_tile/last_tile='1' on the pixel's one beat --
      -- window_m2s_t's own documented contract.
      for p in 0 to 19 loop
        run_pixel(3, 3, 1, false);
      end loop;
      drain_and_check(2000);

    elsif run("test_multi_tile_carry") then
      -- The core of this rewrite: T>1 partial-sum carry across several
      -- consecutive tile beats of one pixel, directed T values first,
      -- then a randomized sweep.
      run_pixel(3, 3, 2, false);
      run_pixel(3, 3, 3, false);
      run_pixel(3, 3, 4, false);
      run_pixel(2, 2, 3, false);
      run_pixel(1, 1, 4, false);
      for p in 0 to 9 loop
        run_pixel(3, 3, 2 + (p mod 3), false);
      end loop;
      drain_and_check(6000);

    elsif run("test_varying_kernels") then
      -- Every c_shapes entry, T=1, including non-square kernels and the
      -- kh*kw*c_tile_channels-not-a-multiple-of-c_pe_cols masking corner
      -- case (this file's header comment).
      for s in c_shapes'range loop
        for p in 0 to 4 loop
          run_pixel(c_shapes(s).h, c_shapes(s).w, 1, false);
        end loop;
      end loop;
      drain_and_check(3000);

    elsif run("test_full_throughput") then
      -- Zero stall on both links (generic-driven, per
      -- module_cnn_accel.py's per-test config): must sustain the ideal
      -- per-pixel cycle count (1 idle accept cycle + num_groups+1 run
      -- cycles = 16, for this fixed 3x3/c_tile_channels=6/c_pe_cols=4
      -- shape) back-to-back, with no extra bubble beyond that.
      start_time := now;
      for p in 0 to 39 loop
        run_pixel(3, 3, 1, p = 39);
      end loop;
      drain_and_check(200);

      check_relation(
        (now - start_time) < 800 * c_clk_period,
        "cnn_accel_pe_array did not sustain full throughput at zero stall"
      );

    elsif run("test_back_to_back_pixels") then
      -- Several pixels streamed one after another (default generic
      -- backpressure), T varying pixel-to-pixel, to prove the per-pixel
      -- weight-row-base restart (first_tile resets to 0) and accumulator
      -- clear never bleed state from the previous pixel into the next.
      for p in 0 to 7 loop
        run_pixel(3, 3, 1 + (p mod 3), p = 7);
      end loop;
      drain_and_check(3000);

    elsif run("test_backpressure") then
      -- Forces a non-zero stall locally on both links regardless of the
      -- generics' own default (see the signal declarations' comment),
      -- exercising stalls mid-tile-sequence (between individual tile
      -- beats of the same pixel), not just between pixels.
      stall_pct_in <= 45;
      stall_pct_out <= 45;
      wait until rising_edge(clk);

      for p in 0 to 14 loop
        run_pixel(3, 3, 1 + (p mod 4), p = 14);
      end loop;
      drain_and_check(6000);

      stall_pct_in <= stall_probability_percent_in;
      stall_pct_out <= stall_probability_percent_out;

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
