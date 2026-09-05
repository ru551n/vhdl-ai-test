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

-- VUnit-5 testbench for cnn_accel_pool. See
-- modules/cnn_accel/doc/cnn_accel_pool_req.md and
-- modules/cnn_accel/doc/cnn_accel_pool_proposal.md section "Verification
-- plan" for the test plan. Hand-rolled record-port stimulus/monitor
-- procedures (no VUnit axi_stream_master/slave VC -- this module's ports
-- are axi_stream_pkg records, not flat t* signals; same practical choice
-- tb_cnn_accel_weight_buffer.vhd already made for its record-typed
-- 's_stream' port).
entity tb_cnn_accel_pool is
  generic (
    -- Split per-link randomized-backpressure generics, swept per test in
    -- module_cnn_accel.py's setup_vunit (0/0/0 for the dedicated
    -- full-throughput test, nonzero otherwise) -- mirrors
    -- module_canny.py's _setup_canny_sobel3x3 precedent (one input,
    -- multiple independently-stalled outputs).
    stall_probability_percent_in : natural := 20;
    stall_probability_percent_max : natural := 20;
    stall_probability_percent_avgsum : natural := 20;
    runner_cfg : string
  );
end entity tb_cnn_accel_pool;

architecture tb of tb_cnn_accel_pool is

  -- Small, directed generics: g_max_kernel_size**2 * 8 = 72 <= 128
  -- (axi_stream_data_sz), well within the module's own contract
  -- (g_max_kernel_size <= 4) -- see proposal doc.
  constant c_kernel_max : positive := 3;
  constant c_max_taps : positive := c_kernel_max * c_kernel_max;
  -- Not a power-of-two multiple/typical 32: proves the reduction is not
  -- silently relying on some other fixed width.
  constant c_accum_width : positive := 20;

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal cfg_opcode : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pool_kernel_h : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pool_kernel_w : std_ulogic_vector(7 downto 0) := (others => '0');

  signal s_window_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_window_s2m : axi_stream_s2m_t;

  signal m_max_m2s : axi_stream_m2s_t;
  signal m_max_s2m : axi_stream_s2m_t := (ready => '0');

  signal m_avgsum_m2s : axi_stream_m2s_t;
  signal m_avgsum_s2m : axi_stream_s2m_t := (ready => '0');

  -- Non-blocking scoreboard: one queue per output port. The stimulus
  -- process pushes each accepted beat's expected (value, last) onto the
  -- queue matching that beat's opcode; the matching monitor process pops
  -- and checks on every accepted output beat. A beat routed to the wrong
  -- port, or a value mismatch, is caught either by a wrong value/last
  -- check or by popping from an unexpectedly-empty queue (VUnit's pop
  -- raises on an empty queue).
  constant max_expected_q : queue_t := new_queue;
  constant avgsum_expected_q : queue_t := new_queue;

  type taps_arr_t is array (0 to c_max_taps - 1) of integer range -128 to 127;

  -- Kernel shapes swept by every test below: 1x1 up to c_kernel_max x
  -- c_kernel_max, square and non-square.
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
  -- Independently re-derived golden model (not calling into the RTL's own
  -- reduce_max/reduce_sum) -- cross-checked by hand against
  -- cnn_accel_model.py's pool_max()/_pool_windows() (sum(taps)) during
  -- authoring: e.g. taps=[-60,-96,2,-68,125,102,113,66,-21] (seed 1, 9
  -- taps) gives max=125, sum=163, matching both this function and the
  -- Python model.
  ------------------------------------------------------------------------

  function golden_max(taps : taps_arr_t; active_count : natural) return integer is
    variable result : integer := -128;
  begin
    for i in 0 to c_max_taps - 1 loop
      if i < active_count and taps(i) > result then
        result := taps(i);
      end if;
    end loop;
    return result;
  end function;

  function golden_sum(taps : taps_arr_t; active_count : natural) return integer is
    variable result : integer := 0;
  begin
    for i in 0 to c_max_taps - 1 loop
      if i < active_count then
        result := result + taps(i);
      end if;
    end loop;
    return result;
  end function;

  -- Packs a tap array into the low bits of an 's_window_m2s.data'-shaped
  -- vector, per the tap-packing convention documented on cnn_accel_pool's
  -- 's_window_m2s' port (row-major, ascending index from the low bits).
  function pack_window(taps : taps_arr_t; active_count : natural) return std_ulogic_vector is
    variable result : std_ulogic_vector(axi_stream_data_sz - 1 downto 0) := (others => '0');
  begin
    for i in 0 to c_max_taps - 1 loop
      if i < active_count then
        result(8 * i + 7 downto 8 * i) := std_ulogic_vector(to_signed(taps(i), 8));
      else
        -- Don't-care lanes: filled with an out-of-range-looking pattern
        -- (not 0) so a bug that accidentally includes them in the
        -- reduction is likely to be caught by the golden-model mismatch.
        result(8 * i + 7 downto 8 * i) := std_ulogic_vector(to_signed(-1, 8));
      end if;
    end loop;
    return result;
  end function;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  -- Structural safety net: the two outputs must be mutually exclusive on
  -- every clock cycle of the whole simulation, not just during the
  -- dedicated mutual-exclusion test -- redundant with (not a substitute
  -- for) the per-port scoreboard's own routing check.
  ------------------------------------------------------------------------

  mutual_exclusion_check : process(clk)
  begin
    if rising_edge(clk) and reset = '0' then
      check_false(
        m_max_m2s.valid = '1' and m_avgsum_m2s.valid = '1',
        "m_max and m_avgsum must never both be valid in the same cycle"
      );
    end if;
  end process;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_pool
    generic map (
      g_max_kernel_size => c_kernel_max,
      g_accum_width => c_accum_width
    )
    port map (
      clk => clk,
      reset => reset,
      cfg_opcode => cfg_opcode,
      cfg_pool_kernel_h => cfg_pool_kernel_h,
      cfg_pool_kernel_w => cfg_pool_kernel_w,
      s_window_m2s => s_window_m2s,
      s_window_s2m => s_window_s2m,
      m_max_m2s => m_max_m2s,
      m_max_s2m => m_max_s2m,
      m_avgsum_m2s => m_avgsum_m2s,
      m_avgsum_s2m => m_avgsum_s2m
    );

  ------------------------------------------------------------------------
  -- Output monitors: independent randomized-'ready' responders, one per
  -- output port. Pop+check the matching queue on every accepted beat.
  ------------------------------------------------------------------------

  monitor_max : process
    variable rnd : RandomPType;
    variable expected_value : signed(7 downto 0);
    variable expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_max_monitor");
    m_max_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < stall_probability_percent_max then
        m_max_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_max_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_max_m2s.valid = '1' and m_max_s2m.ready = '1' then
        expected_value := pop(max_expected_q);
        expected_last := pop(max_expected_q);
        check_equal(m_max_m2s.data(7 downto 0), std_ulogic_vector(expected_value), "m_max data mismatch");
        check_equal(m_max_m2s.last, expected_last, "m_max last mismatch");
      end if;
    end loop;
  end process;

  monitor_avgsum : process
    variable rnd : RandomPType;
    variable expected_value : signed(c_accum_width - 1 downto 0);
    variable expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_avgsum_monitor");
    m_avgsum_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < stall_probability_percent_avgsum then
        m_avgsum_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_avgsum_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_avgsum_m2s.valid = '1' and m_avgsum_s2m.ready = '1' then
        expected_value := pop(avgsum_expected_q);
        expected_last := pop(avgsum_expected_q);
        check_equal(
          m_avgsum_m2s.data(c_accum_width - 1 downto 0), std_ulogic_vector(expected_value),
          "m_avgsum data mismatch"
        );
        check_equal(m_avgsum_m2s.last, expected_last, "m_avgsum last mismatch");
      end if;
    end loop;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;
    variable taps : taps_arr_t;
    variable opcode : std_ulogic_vector(7 downto 0);
    variable beat_idx : natural;
    variable start_time : time;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
    end procedure;

    -- Generates 'count' random taps in -128..127.
    procedure random_taps(p_taps : out taps_arr_t) is
    begin
      for i in 0 to c_max_taps - 1 loop
        p_taps(i) := rnd.RandInt(-128, 127);
      end loop;
    end procedure;

    -- Pushes one 's_window' beat (with randomized input-side stall) and
    -- enqueues its expected result onto the matching scoreboard queue.
    procedure send_beat(
      kernel_h : natural;
      kernel_w : natural;
      p_taps : taps_arr_t;
      p_opcode : std_ulogic_vector(7 downto 0);
      beat_last : std_ulogic
    ) is
      variable active_count : natural;
    begin
      active_count := kernel_h * kernel_w;

      cfg_opcode <= p_opcode;
      cfg_pool_kernel_h <= std_ulogic_vector(to_unsigned(kernel_h, 8));
      cfg_pool_kernel_w <= std_ulogic_vector(to_unsigned(kernel_w, 8));
      s_window_m2s.data <= pack_window(p_taps, active_count);
      s_window_m2s.last <= beat_last;

      if rnd.RandInt(0, 99) < stall_probability_percent_in then
        s_window_m2s.valid <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      s_window_m2s.valid <= '1';
      wait until rising_edge(clk) and s_window_s2m.ready = '1';
      s_window_m2s.valid <= '0';

      if p_opcode = OPCODE_POOL_AVG then
        push(avgsum_expected_q, to_signed(golden_sum(p_taps, active_count), c_accum_width));
        push(avgsum_expected_q, beat_last);
      else
        push(max_expected_q, to_signed(golden_max(p_taps, active_count), 8));
        push(max_expected_q, beat_last);
      end if;
    end procedure;

    -- Random beats across the full kernel-shape sweep, all with the same
    -- fixed opcode.
    procedure run_kernel_sweep(p_opcode : std_ulogic_vector(7 downto 0); beats_per_shape : positive) is
      variable local_taps : taps_arr_t;
    begin
      for s in c_shapes'range loop
        for beat in 0 to beats_per_shape - 1 loop
          random_taps(local_taps);
          send_beat(c_shapes(s).h, c_shapes(s).w, local_taps, p_opcode, '0');
        end loop;
      end loop;
    end procedure;

    -- Directed extremes: every active tap forced to 'value', for every
    -- kernel shape -- exercises the full accumulator width (all-127 /
    -- all--128 sums) and the max identity element (-128 seed) not
    -- accidentally winning over a real, larger tap.
    procedure run_directed_extremes(p_opcode : std_ulogic_vector(7 downto 0); value : integer) is
      variable local_taps : taps_arr_t;
    begin
      for s in c_shapes'range loop
        for i in 0 to c_max_taps - 1 loop
          local_taps(i) := value;
        end loop;
        send_beat(c_shapes(s).h, c_shapes(s).w, local_taps, p_opcode, '0');
      end loop;
    end procedure;

    -- Waits (bounded) until both scoreboard queues have drained, then
    -- confirms they are truly empty (every expected output actually
    -- arrived) rather than just timing out.
    procedure drain_and_check(max_wait_cycles : positive) is
      variable cycles : natural := 0;
    begin
      while (not is_empty(max_expected_q) or not is_empty(avgsum_expected_q)) and cycles < max_wait_cycles loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
      end loop;
      check_true(is_empty(max_expected_q), "m_max scoreboard queue did not drain in time");
      check_true(is_empty(avgsum_expected_q), "m_avgsum scoreboard queue did not drain in time");
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    do_reset;
    wait until rising_edge(clk);

    if run("test_pool_max_kernel_sizes") then
      run_kernel_sweep(OPCODE_POOL_MAX, 15);
      run_directed_extremes(OPCODE_POOL_MAX, -128);
      run_directed_extremes(OPCODE_POOL_MAX, 127);

      -- One beat with both extremes co-present: max must pick 127, not be
      -- confused by -128 also being present.
      for i in 0 to c_max_taps - 1 loop
        if i mod 2 = 0 then
          taps(i) := -128;
        else
          taps(i) := 127;
        end if;
      end loop;
      send_beat(3, 3, taps, OPCODE_POOL_MAX, '1');

      drain_and_check(500);

    elsif run("test_pool_avg_exact_sum") then
      run_kernel_sweep(OPCODE_POOL_AVG, 15);
      run_directed_extremes(OPCODE_POOL_AVG, -128);
      run_directed_extremes(OPCODE_POOL_AVG, 127);
      drain_and_check(500);

    elsif run("test_opcode_mutual_exclusion") then
      -- Back-to-back alternating opcode, fixed 3x3 kernel, minimal
      -- interleave to stress simultaneous overlap possibilities.
      for beat in 0 to 59 loop
        random_taps(taps);
        if beat mod 2 = 0 then
          opcode := OPCODE_POOL_MAX;
        else
          opcode := OPCODE_POOL_AVG;
        end if;
        send_beat(3, 3, taps, opcode, to_sl(beat = 59));
      end loop;
      drain_and_check(500);

    elsif run("test_backpressure") then
      -- Alternating opcode across the full kernel-shape sweep so that
      -- backpressure on *both* m_max and m_avgsum is actually exercised
      -- (not just whichever path happens to be used) -- see proposal doc
      -- "Verification plan".
      beat_idx := 0;
      for s in c_shapes'range loop
        for beat in 0 to 19 loop
          random_taps(taps);
          if beat_idx mod 2 = 0 then
            opcode := OPCODE_POOL_MAX;
          else
            opcode := OPCODE_POOL_AVG;
          end if;
          send_beat(c_shapes(s).h, c_shapes(s).w, taps, opcode, '0');
          beat_idx := beat_idx + 1;
        end loop;
      end loop;
      drain_and_check(2000);

    elsif run("test_full_throughput") then
      -- Zero stall on all three links (generic-driven): must sustain one
      -- output beat per accepted input beat.
      start_time := now;
      for beat in 0 to 299 loop
        random_taps(taps);
        send_beat(3, 3, taps, OPCODE_POOL_MAX, to_sl(beat = 299));
      end loop;
      drain_and_check(350);

      check_relation(
        (now - start_time) < 320 * c_clk_period,
        "cnn_accel_pool did not sustain full throughput at zero stall"
      );

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
