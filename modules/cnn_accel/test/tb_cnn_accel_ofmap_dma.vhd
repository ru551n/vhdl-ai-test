library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.memory_pkg.all;
use vunit_lib.axi_slave_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library bfm;

-- VUnit-5 testbench for cnn_accel_ofmap_dma. See
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_req.md and
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_proposal.md section 8 for the
-- verification plan.
--
-- The AXI write master boundary is served by bfm.axi_write_slave (hdl-modules
-- wrapper around VUnit's axi_write_slave verification component + memory
-- model). Written data is checked byte-exactly by pre-declaring the expected
-- word at each target address ('set_expected_word') and asserting
-- 'check_expected_was_written' at the end of every test. Directed AXI
-- backpressure and held-back BRESPs are produced by switching the BFM's
-- per-channel stall probabilities between 0.0 and 1.0 at runtime.
--
-- VUnit's slave always answers BRESP=OKAY (vunit/vhdl/verification_components/
-- src/axi_write_slave.vhd), so the 'resp_error' test uses a passive
-- wire-level override of the BRESP field between the BFM and the DUT for one
-- chosen B beat -- the BFM still owns every handshake and data byte.
--
-- The producer stream side is driven directly with a small procedure honoring
-- 'ready' backpressure, since several tests interleave cycle-exact checks
-- between individual beats.
entity tb_cnn_accel_ofmap_dma is
  generic (runner_cfg : string);
end entity tb_cnn_accel_ofmap_dma;

architecture tb of tb_cnn_accel_ofmap_dma is

  -- Small, directed generics. 4 bytes/beat keeps addr/length arithmetic
  -- readable while still exercising the alignment contract (proposal doc
  -- section 3.3/6).
  constant c_axi_addr_width : positive := 32;
  constant c_axi_data_width : positive := 32;
  constant c_bytes_per_beat : positive := c_axi_data_width / 8;

  constant c_clk_period : time := 10 ns;
  constant c_settle : time := 1 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '0';

  signal req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal req_s2m : dma_req_s2m_t;
  signal dma_done : std_ulogic;
  signal resp_error : std_ulogic;

  signal s_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_stream_s2m : axi_stream_s2m_t;

  signal m_axi_aw_m2s : axi_m2s_a_t;
  signal m_axi_aw_s2m : axi_s2m_a_t := axi_s2m_a_init;
  signal m_axi_w_m2s : axi_m2s_w_t;
  signal m_axi_w_s2m : axi_s2m_w_t := axi_s2m_w_init;
  signal m_axi_b_m2s : axi_m2s_b_t;
  signal m_axi_b_s2m : axi_s2m_b_t := axi_s2m_b_init;

  -- Bundled view of the DUT's AW+W+B ports for the BFM, and the BFM's own
  -- (always-OKAY) S2M response before the BRESP override below.
  signal axi_write_m2s : axi_write_m2s_t := axi_write_m2s_init;
  signal axi_write_s2m_bfm : axi_write_s2m_t := axi_write_s2m_init;

  -- One flat region covering every address any test below touches (highest
  -- is 2048 + 2 beats). The first allocation in a fresh memory_t starts at
  -- address 0, so DUT addresses map 1:1 onto memory addresses.
  constant c_memory_bytes : positive := 8 * 1024;
  constant memory : memory_t := new_memory;
  -- No randomized stalling by default: several tests make cycle-exact
  -- claims (dma_done not before the last BRESP, ready back the next cycle).
  -- Backpressure is switched on per test via the set_*_stall_probability
  -- procedures. Response FIFO depth > 1 lets more than one AW/W beat be
  -- accepted while their BRESPs are deliberately held back.
  constant axi_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    write_response_fifo_depth => 4
  );

  -- Test-controlled: forces the resp value of the Nth B beat (0-indexed,
  -- counted over the whole test) to 'axi_resp_slverr' instead of
  -- 'axi_resp_okay'. BRESPs are returned in order, so B beat N is the
  -- response to the Nth accepted AW/W beat. -1 = never.
  signal inject_error_on_b_beat : integer := -1;
  signal num_b_beats_served : natural := 0;
  signal num_aw_beats_accepted : natural := 0;

  ------------------------------------------------------------------------
  -- Stream-side helper: push one beat onto 's_stream', honoring 'ready'.
  ------------------------------------------------------------------------

  procedure push_stream_beat(
    signal clk_i : in std_ulogic;
    signal m2s : out axi_stream_m2s_t;
    signal s2m : in axi_stream_s2m_t;
    data_value : in unsigned
  ) is
  begin
    m2s.data <= (others => '0');
    m2s.data(data_value'length - 1 downto 0) <= std_ulogic_vector(data_value);
    m2s.valid <= '1';
    wait until rising_edge(clk_i) and s2m.ready = '1';
    m2s.valid <= '0';
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;

  dut : entity cnn_accel.cnn_accel_ofmap_dma
    generic map (
      g_axi_addr_width => c_axi_addr_width,
      g_axi_data_width => c_axi_data_width
    )
    port map (
      clk => clk,
      reset => reset,
      --
      req_m2s => req_m2s,
      req_s2m => req_s2m,
      dma_done => dma_done,
      resp_error => resp_error,
      --
      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,
      --
      m_axi_aw_m2s => m_axi_aw_m2s,
      m_axi_aw_s2m => m_axi_aw_s2m,
      m_axi_w_m2s => m_axi_w_m2s,
      m_axi_w_s2m => m_axi_w_s2m,
      m_axi_b_m2s => m_axi_b_m2s,
      m_axi_b_s2m => m_axi_b_s2m
    );

  ------------------------------------------------------------------------
  -- AXI4 write slave BFM, backed by the VUnit memory model.
  ------------------------------------------------------------------------
  axi_write_m2s <= (aw => m_axi_aw_m2s, w => m_axi_w_m2s, b => m_axi_b_m2s);
  m_axi_aw_s2m <= axi_write_s2m_bfm.aw;
  m_axi_w_s2m <= axi_write_s2m_bfm.w;

  axi_write_slave_inst : entity bfm.axi_write_slave
    generic map (
      axi_slave => axi_slave,
      data_width => c_axi_data_width,
      id_width => 0,
      address_width => c_axi_addr_width
    )
    port map (
      clk => clk,
      --
      axi_write_m2s => axi_write_m2s,
      axi_write_s2m => axi_write_s2m_bfm
    );

  ------------------------------------------------------------------------
  -- Passive BRESP override: forwards the BFM's B channel untouched except
  -- for the resp field on the single beat whose index equals
  -- 'inject_error_on_b_beat'. 'num_b_beats_served' only advances on a
  -- completed handshake, so the presented resp is stable while valid is
  -- high and not yet accepted.
  ------------------------------------------------------------------------
  bresp_override : process(all)
  begin
    m_axi_b_s2m <= axi_write_s2m_bfm.b;
    if num_b_beats_served = inject_error_on_b_beat then
      m_axi_b_s2m.resp <= axi_resp_slverr;
    end if;
  end process;

  bus_monitor : process
  begin
    wait until rising_edge(clk);
    if m_axi_aw_m2s.valid = '1' and m_axi_aw_s2m.ready = '1' then
      num_aw_beats_accepted <= num_aw_beats_accepted + 1;
    end if;
    if m_axi_b_s2m.valid = '1' and m_axi_b_m2s.ready = '1' then
      num_b_beats_served <= num_b_beats_served + 1;
    end if;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable buf : buffer_t;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      reset <= '0';
    end procedure;

    procedure issue_request(addr_val : natural; length_val : natural) is
    begin
      req_m2s.req.addr <= to_unsigned(addr_val, 32);
      req_m2s.req.length <= to_unsigned(length_val, 32);
      req_m2s.valid <= '1';
      wait until rising_edge(clk) and req_s2m.ready = '1';
      wait for c_settle;
      req_m2s.valid <= '0';
    end procedure;

    -- Declares, in the memory model, the exact words the DUT must write:
    -- 'num_beats' consecutive words from 'base_addr' carrying the counter
    -- pattern starting at 'salt' (the same pattern 'push_beats' produces).
    -- Any write with a different value or to an undeclared address fails
    -- at write time; a missing write fails at 'check_expected_was_written'.
    procedure expect_beats(base_addr : natural; num_beats : natural; salt : natural) is
    begin
      for i in 0 to num_beats - 1 loop
        set_expected_word(
          memory => memory,
          address => base_addr + i * c_bytes_per_beat,
          expected => std_ulogic_vector(to_unsigned(salt + i, c_axi_data_width))
        );
      end loop;
    end procedure;

    -- Pushes 'num_beats' stream beats with a simple counter pattern,
    -- starting at 'salt'.
    procedure push_beats(num_beats : natural; salt : natural) is
    begin
      for i in 0 to num_beats - 1 loop
        push_stream_beat(
          clk, s_stream_m2s, s_stream_s2m,
          to_unsigned(salt + i, c_axi_data_width)
        );
      end loop;
    end procedure;

    procedure wait_for_dma_done is
    begin
      wait until rising_edge(clk) and dma_done = '1';
    end procedure;

    -- Hold the slave's AW and W channels (stall probability 1.0 = never
    -- ready) or release them (0.0 = always ready).
    function stall_probability(stalled : boolean) return real is
    begin
      if stalled then
        return 1.0;
      end if;
      return 0.0;
    end function;

    procedure set_aw_w_stalled(stalled : boolean) is
    begin
      set_address_stall_probability(net, axi_slave, stall_probability(stalled));
      set_data_stall_probability(net, axi_slave, stall_probability(stalled));
    end procedure;

    procedure set_b_stalled(stalled : boolean) is
    begin
      set_write_response_stall_probability(net, axi_slave, stall_probability(stalled));
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    buf := allocate(memory, num_bytes => c_memory_bytes, name => "ofmap_sink");
    check_equal(base_address(buf), 0, "memory region must start at address 0 so DUT addresses map 1:1");

    do_reset;

    if run("test_single_beat_request") then
      -- length = 1 beat: verify AW address, one BRESP=OKAY accepted,
      -- dma_done pulses exactly once, resp_error never, req_s2m.ready
      -- returns high the next cycle.
      issue_request(16, c_bytes_per_beat);
      check_equal(req_s2m.ready, '0', "ready drops the cycle a request is accepted");

      expect_beats(16, 1, 100);
      push_beats(1, 100);
      wait_for_dma_done;
      check_equal(resp_error, '0', "no error on a clean single-beat request");
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready returns high the cycle after dma_done");

    elsif run("test_multi_beat_request") then
      -- length = several beats: verify beat count / dma_done timing
      -- (exactly on the last BRESP, not before).
      issue_request(32, 4 * c_bytes_per_beat);
      expect_beats(32, 4, 200);

      for i in 0 to 2 loop
        push_beats(1, 200 + i);
        check_equal(dma_done, '0', "dma_done must not pulse before the last beat's BRESP");
      end loop;

      push_beats(1, 203);
      wait_for_dma_done;
      check_equal(resp_error, '0', "no error on a clean multi-beat request");

    elsif run("test_back_to_back_requests") then
      -- Second request issued immediately after the first's dma_done:
      -- verify its AW addresses start fresh at its own addr (ring-buffer
      -- reinit actually takes effect) and its own beat count is
      -- independent of the first.
      issue_request(0, 2 * c_bytes_per_beat);
      expect_beats(0, 2, 10);
      push_beats(2, 10);
      wait_for_dma_done;
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready high before issuing the second request");

      issue_request(1000, 3 * c_bytes_per_beat);
      expect_beats(1000, 3, 20);
      push_beats(3, 20);
      wait_for_dma_done;
      check_equal(resp_error, '0', "second request completes cleanly");

    elsif run("test_resp_error_latched_and_reported_at_completion") then
      -- BRESP = SLVERR on the second beat of a 3-beat request: dma_done
      -- must still pulse (full length attempted) and resp_error must
      -- pulse on that same cycle, not immediately when the error occurs.
      inject_error_on_b_beat <= 1;
      issue_request(64, 3 * c_bytes_per_beat);
      expect_beats(64, 3, 300);

      push_beats(1, 300);
      check_equal(resp_error, '0', "resp_error not asserted before request completion");
      push_beats(1, 301);
      check_equal(resp_error, '0', "resp_error still not asserted right after the erroring beat");
      push_beats(1, 302);

      wait until rising_edge(clk) and dma_done = '1';
      check_equal(resp_error, '1', "resp_error pulses on the same cycle as the completing dma_done");

    elsif run("test_stream_backpressure") then
      -- Idle cycles on the producer stream between beats: request must
      -- still complete once the stream resumes, with no protocol
      -- violation in between.
      issue_request(96, 3 * c_bytes_per_beat);
      expect_beats(96, 3, 400);

      push_beats(1, 400);
      for i in 0 to 4 loop
        wait until rising_edge(clk);
      end loop;
      push_beats(1, 401);
      for i in 0 to 4 loop
        wait until rising_edge(clk);
      end loop;
      push_beats(1, 402);

      wait_for_dma_done;
      check_equal(resp_error, '0', "stream-stalled request still completes cleanly");

    elsif run("test_axi_backpressure") then
      -- Slave AW/W ready held low for a few cycles: no protocol violation
      -- (AWVALID held stable while not accepted -- also enforced by the
      -- BFM's own AW/W protocol checkers), request eventually completes.
      issue_request(128, 2 * c_bytes_per_beat);
      expect_beats(128, 2, 500);

      set_aw_w_stalled(true);
      -- Present the beat directly (not via 'push_stream_beat', which
      -- blocks until 's_stream_s2m.ready' -- exactly what must *not*
      -- happen yet here) and hold it at the AXI boundary.
      s_stream_m2s.data <= (others => '0');
      s_stream_m2s.data(c_axi_data_width - 1 downto 0) <=
        std_ulogic_vector(to_unsigned(500, c_axi_data_width));
      s_stream_m2s.valid <= '1';
      -- Beat is presented on 's_stream' and held at the AXI boundary;
      -- AWVALID must stay asserted (checked below) while not accepted.
      for i in 0 to 3 loop
        wait until rising_edge(clk);
        wait for c_settle;
        check_equal(m_axi_aw_m2s.valid, '1', "AWVALID must stay high while not accepted");
      end loop;
      check_equal(num_aw_beats_accepted, 0, "no AW beat may be accepted while the slave is stalled");
      set_aw_w_stalled(false);

      -- Now let the held beat actually complete its handshake before
      -- moving on -- matches what 'push_stream_beat' does internally.
      wait until rising_edge(clk) and s_stream_s2m.ready = '1';
      s_stream_m2s.valid <= '0';

      push_beats(1, 501);
      wait_for_dma_done;
      check_equal(resp_error, '0', "AXI-backpressured request still completes cleanly");

    elsif run("test_abort_mid_request_drains_before_ready_returns") then
      -- Assert reset while at least one BRESP is still outstanding:
      -- req_s2m.ready must drop immediately and return high only once
      -- the pending BRESP(s) have been observed, never before. A fresh
      -- request issued afterwards must behave identically to any other
      -- first request.
      set_b_stalled(true);
      issue_request(256, 2 * c_bytes_per_beat);
      expect_beats(256, 2, 600);
      push_beats(2, 600);

      -- The stream-side handshake completing (above) does not itself mean
      -- the AXI-side AW/W beat has been accepted yet -- there is pipeline
      -- latency between the two inside the wrapped core. Wait for both
      -- beats to actually be accepted before asserting reset, so the
      -- "both accepted, BRESPs still outstanding" premise below actually
      -- holds.
      wait until rising_edge(clk) and num_aw_beats_accepted = 2;
      check_equal(num_b_beats_served, 0, "both BRESPs must still be outstanding");

      -- Both AW/W beats have been accepted (outstanding_q = 2) but their
      -- BRESPs are still held back -- reset now.
      reset <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(req_s2m.ready, '0', "ready must drop immediately on reset with outstanding BRESPs");
      reset <= '0';

      -- Ready must stay low until the pending BRESPs actually drain.
      for i in 0 to 3 loop
        wait until rising_edge(clk);
        wait for c_settle;
        check_equal(req_s2m.ready, '0', "ready must not return before pending BRESPs are observed");
      end loop;

      set_b_stalled(false);
      wait until rising_edge(clk) and req_s2m.ready = '1';
      check_equal(num_b_beats_served, 2, "ready returns only after both pending BRESPs were observed");

      -- A fresh request behaves identically to any other first request
      -- -- no corruption from the aborted one.
      issue_request(2048, 2 * c_bytes_per_beat);
      expect_beats(2048, 2, 700);
      push_beats(2, 700);
      wait_for_dma_done;
      check_equal(resp_error, '0', "post-abort request completes cleanly");

    elsif run("test_abort_with_zero_outstanding_returns_ready_next_cycle") then
      -- Reset while idle (no outstanding transactions): ready returns
      -- high the very next cycle, no spurious s_drain detour.
      check_equal(req_s2m.ready, '1', "ready high before this test's reset");
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready returns high the cycle after an idle-time reset");

    elsif run("test_zero_length_request") then
      -- length = 0: dma_done pulses with no AW/W activity at all,
      -- req_s2m.ready returns high one cycle later.
      issue_request(512, 0);
      wait until rising_edge(clk) and dma_done = '1';
      check_equal(m_axi_aw_m2s.valid, '0', "no AW activity for a zero-length request");
      check_equal(m_axi_w_m2s.valid, '0', "no W activity for a zero-length request");
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready returns high the cycle after the zero-length dma_done");
    end if;

    -- Every word declared with 'expect_beats' must have been written with
    -- exactly that value; nothing else may have been written.
    check_expected_was_written(memory);

    test_runner_cleanup(runner);
    wait;
  end process;

  test_runner_watchdog(runner, 1 ms);

end architecture;
