library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- VUnit-5 testbench for cnn_accel_ofmap_dma. See
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_req.md and
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_proposal.md section 8 for the
-- verification plan.
--
-- No VUnit AXI4/AXI4-Stream VC is used for the AXI write master boundary:
-- this testbench needs exact, directed control over 'BRESP' values and
-- timing (to exercise 'resp_error' latching and the reset/'s_drain' abort
-- path -- proposal doc section 4/8), which the standard 'bfm.axi_write_slave'
-- VC does not expose. A small hand-written AXI write slave process plays
-- that role instead, following the same "hand-written non-blocking helper"
-- precedent as tb_cnn_accel_weight_buffer's fill-side driver. The producer
-- stream side is likewise driven directly with a small procedure, honoring
-- 'ready' backpressure.
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

  ------------------------------------------------------------------------
  -- Hand-written AXI write slave state -- see header comment above.
  ------------------------------------------------------------------------

  -- Both AW and W are always presented (and therefore accepted) on the
  -- same cycle here: the wrapped 'dma_axi_write_simple' core's single-
  -- beat-packet implementation merges 'segment_valid'/'axi_valid' into one
  -- combined handshake before splitting it back out to 'aw'/'w'
  -- (hdl-modules 'common.handshake_merger'/'handshake_splitter'), so tying
  -- both readies to the same control signal is sufficient and keeps this
  -- slave simple.
  signal slave_ready_ctrl : std_ulogic := '1';

  constant c_max_pending_b : positive := 64;
  type resp_array_t is array (0 to c_max_pending_b - 1) of axi_resp_t;

  signal aw_capture_addr : unsigned(c_axi_addr_width - 1 downto 0);
  signal aw_capture_valid : std_ulogic := '0';
  signal w_capture_data : std_ulogic_vector(c_axi_data_width - 1 downto 0);

  -- Per-beat trace of the accepted AW address / W data, indexed by the
  -- (pre-increment) 'captured_beat_count' at accept time, so tests can
  -- assert byte-exact addresses/data rather than only handshake timing.
  -- Driven by 'b_bfm' alongside 'captured_beat_count' -- single driver.
  type addr_trace_t is array (0 to c_max_pending_b - 1) of unsigned(c_axi_addr_width - 1 downto 0);
  type data_trace_t is array (0 to c_max_pending_b - 1) of std_ulogic_vector(c_axi_data_width - 1 downto 0);

  signal captured_addr_trace : addr_trace_t;
  signal captured_data_trace : data_trace_t;

  -- Test-controlled: forces the resp value of the Nth accepted beat
  -- (0-indexed, counting from the last 'reset_capture_counters' call) to
  -- 'axi_resp_slverr' instead of 'axi_resp_okay'. -1 = never.
  signal inject_error_on_beat : integer := -1;
  -- Test-controlled: number of idle cycles between a beat's AW/W accept
  -- and its BRESP becoming valid. 0 = same cycle.
  signal b_response_delay_cycles : natural := 0;
  -- Test-controlled: pulsed for one cycle by 'reset_capture_counters' to
  -- reset 'captured_beat_count' back to zero. Kept as a separate signal
  -- (rather than driving 'captured_beat_count' directly from the 'main'
  -- process) so that 'captured_beat_count' has a single driver: the
  -- 'b_bfm' process below.
  signal capture_reset_req : std_ulogic := '0';

  signal captured_beat_count : natural := 0;

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

  m_axi_aw_s2m.ready <= slave_ready_ctrl;
  m_axi_w_s2m.ready <= slave_ready_ctrl;

  aw_capture_addr <= m_axi_aw_m2s.addr(c_axi_addr_width - 1 downto 0);
  w_capture_data <= m_axi_w_m2s.data(c_axi_data_width - 1 downto 0);
  aw_capture_valid <=
    '1' when slave_ready_ctrl = '1' and m_axi_aw_m2s.valid = '1' and m_axi_w_m2s.valid = '1' else
    '0';

  ------------------------------------------------------------------------
  -- B-response state machine: one pending-response FIFO entry pushed per
  -- accepted AW/W beat, popped (after 'b_response_delay_cycles' idle
  -- cycles) one at a time, in order -- matches the single-outstanding-ID
  -- ordering rule this module's own design relies on (proposal doc
  -- section 4, point 6).
  ------------------------------------------------------------------------

  b_bfm : process(clk)
    variable jobs : resp_array_t;
    variable head : natural := 0;
    variable tail : natural := 0;
    variable count : natural := 0;
    variable delay_remaining : natural := 0;
    variable b_active : boolean := false;
    variable current_resp : axi_resp_t := axi_resp_okay;
  begin
    if rising_edge(clk) then
      if aw_capture_valid = '1' then
        if count = 0 then
          -- Queue was empty (drained, or this is the very first beat):
          -- seed the delay for this newly-arriving beat too, not just
          -- for beats popped after an earlier one -- otherwise a
          -- 'b_response_delay_cycles' set up before the first beat of a
          -- request is silently ignored for that first beat.
          delay_remaining := b_response_delay_cycles;
        end if;
        if captured_beat_count = inject_error_on_beat then
          jobs(tail) := axi_resp_slverr;
        else
          jobs(tail) := axi_resp_okay;
        end if;
        tail := (tail + 1) mod c_max_pending_b;
        count := count + 1;
        captured_addr_trace(captured_beat_count) <= aw_capture_addr;
        captured_data_trace(captured_beat_count) <= w_capture_data;
        captured_beat_count <= captured_beat_count + 1;
      end if;

      if capture_reset_req = '1' then
        captured_beat_count <= 0;
      end if;

      if not b_active then
        m_axi_b_s2m.valid <= '0';

        if count > 0 then
          if delay_remaining = 0 then
            b_active := true;
            current_resp := jobs(head);
            m_axi_b_s2m.valid <= '1';
            m_axi_b_s2m.resp <= jobs(head);
          else
            delay_remaining := delay_remaining - 1;
          end if;
        end if;
      else
        if m_axi_b_m2s.ready = '1' then
          head := (head + 1) mod c_max_pending_b;
          count := count - 1;
          b_active := false;
          delay_remaining := b_response_delay_cycles;
          m_axi_b_s2m.valid <= '0';
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  main : process
    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      reset <= '0';
    end procedure;

    procedure reset_capture_counters is
    begin
      capture_reset_req <= '1';
      inject_error_on_beat <= -1;
      wait until rising_edge(clk);
      wait for c_settle;
      capture_reset_req <= '0';
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

    -- Byte-exact check of the 'idx'th (0-indexed, since the last
    -- 'reset_capture_counters' call) accepted AW address / W data
    -- against the values this request/producer must have driven.
    procedure check_capture(idx : natural; expected_addr_val : natural; expected_data_val : natural) is
    begin
      check_equal(
        captured_addr_trace(idx), to_unsigned(expected_addr_val, c_axi_addr_width),
        "beat " & to_string(idx) & " AW address mismatch"
      );
      check_equal(
        captured_data_trace(idx), std_ulogic_vector(to_unsigned(expected_data_val, c_axi_data_width)),
        "beat " & to_string(idx) & " W data mismatch"
      );
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    do_reset;
    reset_capture_counters;

    if run("test_single_beat_request") then
      -- length = 1 beat: verify AW address, one BRESP=OKAY accepted,
      -- dma_done pulses exactly once, resp_error never, req_s2m.ready
      -- returns high the next cycle.
      issue_request(16, c_bytes_per_beat);
      check_equal(req_s2m.ready, '0', "ready drops the cycle a request is accepted");

      push_beats(1, 100);
      wait_for_dma_done;
      check_equal(resp_error, '0', "no error on a clean single-beat request");
      check_capture(0, 16, 100);
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready returns high the cycle after dma_done");

    elsif run("test_multi_beat_request") then
      -- length = several beats: verify beat count / dma_done timing
      -- (exactly on the last BRESP, not before).
      issue_request(32, 4 * c_bytes_per_beat);

      for i in 0 to 2 loop
        push_beats(1, 200 + i);
        check_equal(dma_done, '0', "dma_done must not pulse before the last beat's BRESP");
      end loop;

      push_beats(1, 203);
      wait_for_dma_done;
      check_equal(resp_error, '0', "no error on a clean multi-beat request");
      check_capture(0, 32, 200);
      check_capture(1, 36, 201);
      check_capture(2, 40, 202);
      check_capture(3, 44, 203);

    elsif run("test_back_to_back_requests") then
      -- Second request issued immediately after the first's dma_done:
      -- verify its AW addresses start fresh at its own addr (ring-buffer
      -- reinit actually takes effect) and its own beat count is
      -- independent of the first.
      issue_request(0, 2 * c_bytes_per_beat);
      push_beats(2, 10);
      wait_for_dma_done;
      check_capture(0, 0, 10);
      check_capture(1, 4, 11);
      wait for c_settle;
      check_equal(req_s2m.ready, '1', "ready high before issuing the second request");

      issue_request(1000, 3 * c_bytes_per_beat);
      push_beats(3, 20);
      wait_for_dma_done;
      check_equal(resp_error, '0', "second request completes cleanly");
      check_capture(2, 1000, 20);
      check_capture(3, 1004, 21);
      check_capture(4, 1008, 22);

    elsif run("test_resp_error_latched_and_reported_at_completion") then
      -- BRESP = SLVERR on the second beat of a 3-beat request: dma_done
      -- must still pulse (full length attempted) and resp_error must
      -- pulse on that same cycle, not immediately when the error occurs.
      inject_error_on_beat <= 1;
      issue_request(64, 3 * c_bytes_per_beat);

      push_beats(1, 300);
      check_equal(resp_error, '0', "resp_error not asserted before request completion");
      push_beats(1, 301);
      check_equal(resp_error, '0', "resp_error still not asserted right after the erroring beat");
      push_beats(1, 302);

      wait until rising_edge(clk) and dma_done = '1';
      check_equal(resp_error, '1', "resp_error pulses on the same cycle as the completing dma_done");
      check_capture(0, 64, 300);
      check_capture(1, 68, 301);
      check_capture(2, 72, 302);

    elsif run("test_stream_backpressure") then
      -- Idle cycles on the producer stream between beats: request must
      -- still complete once the stream resumes, with no protocol
      -- violation in between.
      issue_request(96, 3 * c_bytes_per_beat);

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
      check_capture(0, 96, 400);
      check_capture(1, 100, 401);
      check_capture(2, 104, 402);

    elsif run("test_axi_backpressure") then
      -- Slave AW/W ready held low for a few cycles: no protocol
      -- violation (AWVALID held stable while not accepted), request
      -- eventually completes.
      issue_request(128, 2 * c_bytes_per_beat);

      slave_ready_ctrl <= '0';
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
      slave_ready_ctrl <= '1';

      -- Now let the held beat actually complete its handshake before
      -- moving on -- matches what 'push_stream_beat' does internally.
      wait until rising_edge(clk) and s_stream_s2m.ready = '1';
      s_stream_m2s.valid <= '0';

      push_beats(1, 501);
      wait_for_dma_done;
      check_equal(resp_error, '0', "AXI-backpressured request still completes cleanly");
      check_capture(0, 128, 500);
      check_capture(1, 132, 501);

    elsif run("test_abort_mid_request_drains_before_ready_returns") then
      -- Assert reset while at least one BRESP is still outstanding:
      -- req_s2m.ready must drop immediately and return high only once
      -- the pending BRESP(s) have been observed, never before. A fresh
      -- request issued afterwards must behave identically to any other
      -- first request.
      b_response_delay_cycles <= 5;
      issue_request(256, 2 * c_bytes_per_beat);
      push_beats(2, 600);

      -- The stream-side handshake completing (above) does not itself mean
      -- the AXI-side AW/W beat has been accepted yet -- there is pipeline
      -- latency between the two inside the wrapped core. Wait for both
      -- beats to actually be captured before asserting reset, so the
      -- "both accepted, BRESPs still outstanding" premise below actually
      -- holds.
      wait until rising_edge(clk) and captured_beat_count = 2;
      check_capture(0, 256, 600);
      check_capture(1, 260, 601);

      -- Both AW/W beats have been accepted (outstanding_q = 2) but their
      -- BRESPs are still delayed -- reset now.
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

      wait until rising_edge(clk) and req_s2m.ready = '1';
      b_response_delay_cycles <= 0;

      -- A fresh request behaves identically to any other first request
      -- -- no corruption from the aborted one.
      issue_request(2048, 2 * c_bytes_per_beat);
      push_beats(2, 700);
      wait_for_dma_done;
      check_equal(resp_error, '0', "post-abort request completes cleanly");
      check_capture(2, 2048, 700);
      check_capture(3, 2052, 701);

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

    test_runner_cleanup(runner);
    wait;
  end process;

  test_runner_watchdog(runner, 1 ms);

end architecture;
