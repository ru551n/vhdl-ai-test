library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- VUnit-5 testbench for cnn_accel_tensor_mem.
--
-- This is the regression test for the read-path data-loss defect fixed in
-- cnn_accel_tensor_mem.vhd's read path (see that file's read_arbitrate
-- header comment for the full writeup). Before the fix, a channel that
-- drained its output register on cycle T re-issued a bank read on cycle T
-- as well as T+1, and the T beat was silently overwritten in the per-bank
-- read register before the consumer ever saw it -- one dropped scratchpad
-- beat per consumer stall, with no error and no assertion. This testbench
-- previously did not exist at all, which is how the bug reached integration
-- (10/14 tb_cnn_accel_top tests broken) before being caught here.
--
-- The current, fixed design makes losslessness a property of the DUT's data
-- path rather than of its issue timing: each read channel has a private
-- two-deep buffer (an output register plus a landing/"skid" register), so a
-- beat returning from the bank while the output register is still occupied
-- has a second slot to land in instead of overwriting anything. The issue
-- rule ('rN_can_issue') only has to prove there is room for the *next*
-- returning beat before granting a new read; it no longer has to serialize
-- issue and capture the way the earlier throttling fix did (see below), so
-- the channel can keep a read in flight across a consumer stall and sustain
-- one beat per cycle.
--
-- Write side: driven directly through 'w0' with a small backpressure-
-- honoring push procedure (module has only one physical write port per
-- bank and the write FSMs are simple counters, so no VUnit VC is needed --
-- matches tb_cnn_accel_weight_buffer.vhd's/tb_cnn_accel_ofmap_dma.vhd's
-- identical choice for this project's own record-typed handshake links).
--
-- Read side: each of 'r0'/'r1' has its own free-running consumer process
-- that drives 'mN_rN_s2m.ready' according to a live 'rN_mode' signal (full
-- random stall, a directed drain-then-stall toggle -- the exact T/T+1
-- pattern that triggers the original bug -- or held-high full rate) and
-- pushes every accepted beat's data/last into a per-channel queue. The main
-- process issues 'rN_req'/'wN_req' DMA requests and then drains/checks
-- those queues once 'rN_done' pulses.
entity tb_cnn_accel_tensor_mem is
  generic (runner_cfg : string);
end entity tb_cnn_accel_tensor_mem;

architecture tb of tb_cnn_accel_tensor_mem is

  -- Small, directed generics. g_bank_words must be a power of two (DUT
  -- assertion); kept small so a full-bank sweep (throughput test) still
  -- simulates fast.
  constant c_num_banks : positive := 2;
  constant c_bank_words : positive := 64;
  constant c_data_width : positive := 32;
  constant c_bytes_per_word : positive := c_data_width / 8;

  constant c_clk_period : time := 10 ns;
  constant c_settle : time := 1 ns;

  -- Read-consumer modes, driven per test via 'r0_mode'/'r1_mode'.
  constant c_mode_random : natural := 0;
  constant c_mode_toggle : natural := 1;
  constant c_mode_full_speed : natural := 2;

  -- Fixed read-pipeline latency, in cycles, from the edge at which a read
  -- request is accepted (which is where the throughput tests below start
  -- their clock) to the edge at which the first beat of that request
  -- appears on 'm_rN_m2s': one cycle of synchronous bank-RAM read latency
  -- plus one cycle for the channel's output register. This is pipeline
  -- fill, not throughput: it is a constant regardless of transfer length,
  -- so the throughput tests subtract it before checking the sustained
  -- rate, and check it separately (by using it as an exact allowance) so
  -- that a latency regression still fails them.
  constant c_read_pipeline_cycles : natural := 2;

  -- test_r0_no_refill_bubble_after_stall: how many cycles into the burst
  -- (from request acceptance) to hold the pause, and for how many cycles.
  -- 'c_stall_start_cycles' must comfortably exceed 'c_read_pipeline_cycles'
  -- so the pipeline has already reached steady state before the pause.
  constant c_stall_start_cycles : natural := 10;
  constant c_stall_len : natural := 5;

  -- test_r0_no_refill_bubble_after_stall: word count for r0's own range
  -- (bank 0, offset 0..c_bubble_r0_words-1); r1 contends for the rest of
  -- the same bank (offset c_bubble_r0_words..c_bank_words-1).
  constant c_bubble_r0_words : natural := 20;

  -- Exact expected cycle span for r0's transfer in
  -- test_r0_no_refill_bubble_after_stall, determined empirically (the
  -- pattern is fully deterministic): pipeline fill, 'c_bubble_r0_words'
  -- beats each costing (on average) 2 contended bank cycles because r1
  -- shares the bank continuously, plus the deliberate 'c_stall_len'-cycle
  -- stall. A refill bubble after the stall adds cycles beyond this.
  constant c_bubble_expected_span_cycles : natural := 43;

  -- test_write_w0_w1_same_bank_round_robin: word count per channel (both
  -- disjoint offset ranges within bank 0, so a subsequent read can check
  -- each channel's data independently).
  constant c_write_contend_words : natural := 20;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '0';

  signal w0_req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal w0_req_s2m : dma_req_s2m_t;
  signal s_w0_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_w0_s2m : axi_stream_s2m_t;
  signal w0_done : std_ulogic;

  signal w1_req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal w1_req_s2m : dma_req_s2m_t;
  signal s_w1_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_w1_s2m : axi_stream_s2m_t;
  signal w1_done : std_ulogic;

  signal r0_req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal r0_req_s2m : dma_req_s2m_t;
  signal m_r0_m2s : axi_stream_m2s_t;
  signal m_r0_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal r0_done : std_ulogic;

  signal r1_req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal r1_req_s2m : dma_req_s2m_t;
  signal m_r1_m2s : axi_stream_m2s_t;
  signal m_r1_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal r1_done : std_ulogic;

  -- Live read-consumer controls -- see the two consumer processes below.
  signal r0_mode : natural := c_mode_random;
  signal r1_mode : natural := c_mode_random;
  signal r0_stall_pct : natural := 40;
  signal r1_stall_pct : natural := 40;

  -- Directed mid-burst pause for c_mode_full_speed on r0 only, used by
  -- test_r0_no_refill_bubble_after_stall (see that test for why). Ignored
  -- by every other mode/test; defaults to never pausing.
  signal r0_pause : std_ulogic := '0';

  -- Per-channel captured-beat queues: one entry per accepted output beat,
  -- pushed as (data, last) pairs, popped/checked by the main process.
  constant r0_captured_q : queue_t := new_queue;
  constant r1_captured_q : queue_t := new_queue;

  ------------------------------------------------------------------------
  -- Deterministic write pattern: word 'i' (0-based, within one request) of
  -- a request started with 'salt' carries value 'salt + i', wrapped to
  -- 'c_data_width' bits.
  ------------------------------------------------------------------------

  -- No 'mod 2**c_data_width' here: with c_data_width=32 that exponent
  -- itself overflows the (32-bit signed) 'natural' range this function
  -- would need to compute it in, and every 'salt + i' this testbench ever
  -- forms is tiny (well under 2**c_data_width) anyway, so 'to_unsigned'
  -- alone is exact.
  function word_value(salt : natural; i : natural) return std_ulogic_vector is
  begin
    return std_ulogic_vector(to_unsigned(salt + i, c_data_width));
  end function;

  function word_addr_bytes(bank : natural; offset : natural) return natural is
  begin
    return (bank * c_bank_words + offset) * c_bytes_per_word;
  end function;

  ------------------------------------------------------------------------
  -- Write-side helper: push one beat onto 's_wN', honoring 'ready'.
  ------------------------------------------------------------------------

  procedure push_write_beat(
    signal clk_i : in std_ulogic;
    signal m2s : out axi_stream_m2s_t;
    signal s2m : in axi_stream_s2m_t;
    data_value : in std_ulogic_vector
  ) is
  begin
    m2s.data <= (others => '0');
    m2s.data(data_value'length - 1 downto 0) <= data_value;
    m2s.valid <= '1';
    wait until rising_edge(clk_i) and s2m.ready = '1';
    m2s.valid <= '0';
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;

  dut : entity cnn_accel.cnn_accel_tensor_mem
    generic map (
      g_num_banks => c_num_banks,
      g_bank_words => c_bank_words,
      g_data_width => c_data_width
    )
    port map (
      clk => clk,
      reset => reset,
      --
      w0_req_m2s => w0_req_m2s,
      w0_req_s2m => w0_req_s2m,
      s_w0_m2s => s_w0_m2s,
      s_w0_s2m => s_w0_s2m,
      w0_done => w0_done,
      --
      w1_req_m2s => w1_req_m2s,
      w1_req_s2m => w1_req_s2m,
      s_w1_m2s => s_w1_m2s,
      s_w1_s2m => s_w1_s2m,
      w1_done => w1_done,
      --
      r0_req_m2s => r0_req_m2s,
      r0_req_s2m => r0_req_s2m,
      m_r0_m2s => m_r0_m2s,
      m_r0_s2m => m_r0_s2m,
      r0_done => r0_done,
      --
      r1_req_m2s => r1_req_m2s,
      r1_req_s2m => r1_req_s2m,
      m_r1_m2s => m_r1_m2s,
      m_r1_s2m => m_r1_s2m,
      r1_done => r1_done
    );

  ------------------------------------------------------------------------
  -- Read channel 0 consumer: free-running for the whole simulation,
  -- behavior selected by 'r0_mode'.
  --
  -- c_mode_toggle is the directed pattern that reproduces the original bug
  -- deterministically: ready high for exactly one cycle (drains the output
  -- register), then low for exactly one cycle (a stall on the very next
  -- cycle) -- exactly the T/T+1 sequence described in
  -- cnn_accel_tensor_mem.vhd's read_arbitrate header comment. On a design
  -- without the landing/skid register (or with an issue rule that lets a
  -- second beat land while the first is still unaccepted and the skid slot
  -- is unavailable), this pattern loses a beat on every single drain, so it
  -- fails fast and deterministically rather than relying on random luck.
  ------------------------------------------------------------------------

  consume_r0 : process
    variable rnd : RandomPType;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_r0");
    m_r0_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      case r0_mode is
        when c_mode_toggle =>
          m_r0_s2m.ready <= '1';
          wait until rising_edge(clk);
          if m_r0_m2s.valid = '1' and m_r0_s2m.ready = '1' then
            push(r0_captured_q, m_r0_m2s.data(c_data_width - 1 downto 0));
            push(r0_captured_q, m_r0_m2s.last);
          end if;
          m_r0_s2m.ready <= '0';
          wait until rising_edge(clk);

        when c_mode_full_speed =>
          -- 'r0_pause' lets a directed test hold this channel's 'ready' low
          -- for a controlled number of cycles mid-burst without switching
          -- away from full-speed mode; every accepted beat's time is also
          -- recorded so that test can check the inter-beat spacing around
          -- the pause for a post-stall refill bubble.
          if r0_pause = '1' then
            m_r0_s2m.ready <= '0';
          else
            m_r0_s2m.ready <= '1';
          end if;
          wait until rising_edge(clk);
          if m_r0_m2s.valid = '1' and m_r0_s2m.ready = '1' then
            push(r0_captured_q, m_r0_m2s.data(c_data_width - 1 downto 0));
            push(r0_captured_q, m_r0_m2s.last);
          end if;

        when others => -- c_mode_random
          if rnd.RandInt(0, 99) < r0_stall_pct then
            m_r0_s2m.ready <= '0';
            for i in 1 to rnd.RandInt(1, 8) loop
              -- Abandon a multi-cycle random stall as soon as the main
              -- process switches mode, so 'r0_mode' takes effect on the
              -- next edge rather than up to 8 cycles later. Without this,
              -- a test that switches to c_mode_full_speed could start
              -- measuring while the consumer is still stalling, and read
              -- that TB-side stall as a DUT throughput bubble.
              exit when r0_mode /= c_mode_random;
              wait until rising_edge(clk);
            end loop;
          end if;
          m_r0_s2m.ready <= '1';
          wait until rising_edge(clk);
          if m_r0_m2s.valid = '1' and m_r0_s2m.ready = '1' then
            push(r0_captured_q, m_r0_m2s.data(c_data_width - 1 downto 0));
            push(r0_captured_q, m_r0_m2s.last);
          end if;
      end case;
    end loop;
  end process;

  -- Read channel 1 consumer: identical structure to 'consume_r0', its own
  -- seed/mode/stall/queue.
  consume_r1 : process
    variable rnd : RandomPType;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_r1");
    m_r1_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      case r1_mode is
        when c_mode_toggle =>
          m_r1_s2m.ready <= '1';
          wait until rising_edge(clk);
          if m_r1_m2s.valid = '1' and m_r1_s2m.ready = '1' then
            push(r1_captured_q, m_r1_m2s.data(c_data_width - 1 downto 0));
            push(r1_captured_q, m_r1_m2s.last);
          end if;
          m_r1_s2m.ready <= '0';
          wait until rising_edge(clk);

        when c_mode_full_speed =>
          m_r1_s2m.ready <= '1';
          wait until rising_edge(clk);
          if m_r1_m2s.valid = '1' and m_r1_s2m.ready = '1' then
            push(r1_captured_q, m_r1_m2s.data(c_data_width - 1 downto 0));
            push(r1_captured_q, m_r1_m2s.last);
          end if;

        when others => -- c_mode_random
          if rnd.RandInt(0, 99) < r1_stall_pct then
            m_r1_s2m.ready <= '0';
            for i in 1 to rnd.RandInt(1, 8) loop
              -- Abandon a multi-cycle random stall as soon as the main
              -- process switches mode, so 'r1_mode' takes effect on the
              -- next edge rather than up to 8 cycles later. Without this,
              -- a test that switches to c_mode_full_speed could start
              -- measuring while the consumer is still stalling, and read
              -- that TB-side stall as a DUT throughput bubble.
              exit when r1_mode /= c_mode_random;
              wait until rising_edge(clk);
            end loop;
          end if;
          m_r1_s2m.ready <= '1';
          wait until rising_edge(clk);
          if m_r1_m2s.valid = '1' and m_r1_s2m.ready = '1' then
            push(r1_captured_q, m_r1_m2s.data(c_data_width - 1 downto 0));
            push(r1_captured_q, m_r1_m2s.last);
          end if;
      end case;
    end loop;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable t_start, t_end : time;
    variable t_start_r1, t_end_r1 : time;
    variable r0_seen, r1_seen : boolean;

    -- test_write_w0_w1_same_bank_round_robin bookkeeping.
    variable wr_w0_left, wr_w1_left : natural;
    variable wr_w0_idx, wr_w1_idx : natural;
    variable wr_both_wanted : boolean;
    variable wr_w0_won, wr_w1_won : boolean;
    variable wr_prev_grant : natural range 0 to 1;
    variable wr_have_prev : boolean;
    variable wr_w0_done_seen, wr_w1_done_seen : boolean;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      reset <= '0';
      -- Flush any stray captures the consumers made while unreset (none
      -- expected, but keeps every test's queues starting empty).
      flush(r0_captured_q);
      flush(r1_captured_q);
    end procedure;

    -- Issues one write request on w0/w1 and streams 'num_words' beats
    -- starting at 'salt', blocking until the last beat is accepted.
    procedure write_words(
      signal wr_req_m2s : out dma_req_m2s_t;
      signal wr_req_s2m : in dma_req_s2m_t;
      signal wr_m2s : out axi_stream_m2s_t;
      signal wr_s2m : in axi_stream_s2m_t;
      bank : natural;
      offset : natural;
      num_words : natural;
      salt : natural
    ) is
    begin
      wr_req_m2s.req.addr <= to_unsigned(word_addr_bytes(bank, offset), 32);
      wr_req_m2s.req.length <= to_unsigned(num_words * c_bytes_per_word, 32);
      wr_req_m2s.valid <= '1';
      wait until rising_edge(clk) and wr_req_s2m.ready = '1';
      wait for c_settle;
      wr_req_m2s.valid <= '0';

      for i in 0 to num_words - 1 loop
        push_write_beat(clk, wr_m2s, wr_s2m, word_value(salt, i));
      end loop;
    end procedure;

    -- Issues one write request on wr_req_*, non-blocking on request
    -- acceptance (mirrors 'issue_read' below) so the caller can drive both
    -- write channels' beats concurrently afterward instead of streaming one
    -- channel's whole transfer before starting the other.
    procedure issue_write(
      signal wr_req_m2s : out dma_req_m2s_t;
      signal wr_req_s2m : in dma_req_s2m_t;
      bank : natural;
      offset : natural;
      num_words : natural
    ) is
    begin
      wr_req_m2s.req.addr <= to_unsigned(word_addr_bytes(bank, offset), 32);
      wr_req_m2s.req.length <= to_unsigned(num_words * c_bytes_per_word, 32);
      wr_req_m2s.valid <= '1';
      wait until rising_edge(clk) and wr_req_s2m.ready = '1';
      wait for c_settle;
      wr_req_m2s.valid <= '0';
    end procedure;

    -- Issues one read request on rN, non-blocking on request acceptance so
    -- the caller can issue r0 and r1 back-to-back without waiting a whole
    -- request out (used by the concurrency test).
    procedure issue_read(
      signal rd_req_m2s : out dma_req_m2s_t;
      signal rd_req_s2m : in dma_req_s2m_t;
      bank : natural;
      offset : natural;
      num_words : natural
    ) is
    begin
      rd_req_m2s.req.addr <= to_unsigned(word_addr_bytes(bank, offset), 32);
      rd_req_m2s.req.length <= to_unsigned(num_words * c_bytes_per_word, 32);
      rd_req_m2s.valid <= '1';
      wait until rising_edge(clk) and rd_req_s2m.ready = '1';
      wait for c_settle;
      rd_req_m2s.valid <= '0';
    end procedure;

    -- Pops 'num_words' (data, last) pairs from 'q' and checks them against
    -- the 'salt'-tagged pattern, in order; also checks the queue holds
    -- exactly that many entries (no extra/leftover beats) and that only
    -- the final one carries 'last'. This is what would have caught the
    -- original bug: a dropped beat shows up either as a straight data
    -- mismatch (a later word replacing an earlier one) or as the queue
    -- coming up short.
    procedure check_captured(q : queue_t; num_words : natural; salt : natural; msg : string) is
      variable data_v : std_ulogic_vector(c_data_width - 1 downto 0);
      variable last_v : std_ulogic;
    begin
      for i in 0 to num_words - 1 loop
        check_false(is_empty(q), msg & ": beat " & natural'image(i) & " missing (queue empty -- beat lost)");
        data_v := pop(q);
        last_v := pop(q);
        check_equal(data_v, word_value(salt, i), msg & ": beat " & natural'image(i) & " data");
        if i = num_words - 1 then
          check_equal(last_v, '1', msg & ": final beat must carry 'last'");
        else
          check_equal(last_v, '0', msg & ": non-final beat must not carry 'last'");
        end if;
      end loop;
      check_true(is_empty(q), msg & ": no extra/duplicate beats delivered");
    end procedure;

    -- Throughput check for one non-backpressured read of 'num_words'
    -- beats. 't_from' must be the time of the edge on which the read
    -- request was accepted (which is where 'issue_read' leaves the
    -- caller) and 't_to' the time of the edge on which that channel's
    -- 'done' was observed, i.e. one edge after the last beat's cycle.
    --
    -- The measured span is therefore 'c_read_pipeline_cycles' of fixed
    -- pipeline fill plus one cycle per beat presented. Requiring
    --
    --   span <= num_words + c_read_pipeline_cycles
    --
    -- is exactly "every beat after the first came out on the cycle
    -- immediately after its predecessor" -- zero bubbles, a sustained 1
    -- beat/cycle -- and, because the allowance is the exact fill latency
    -- rather than a slack margin, it also fails if that latency grows.
    procedure check_full_rate(t_from : time; t_to : time; num_words : natural; msg : string) is
      variable span_cycles : natural;
      variable stream_cycles : natural;
    begin
      span_cycles := (t_to - t_from) / c_clk_period;
      stream_cycles := span_cycles - c_read_pipeline_cycles;
      info(
        msg & ": " & natural'image(num_words) & " beats streamed in " &
        natural'image(stream_cycles) & " cycles (" &
        real'image(real(num_words) / real(stream_cycles)) &
        " beats/cycle) after " & natural'image(c_read_pipeline_cycles) &
        " cycles of pipeline fill; target is 1.0 beats/cycle"
      );
      check_relation(
        span_cycles <= num_words + c_read_pipeline_cycles,
        msg & ": does not sustain 1 beat/cycle (measured " &
        natural'image(stream_cycles) & " cycles for " & natural'image(num_words) &
        " beats, allowing " & natural'image(c_read_pipeline_cycles) &
        " cycles of pipeline fill)"
      );
    end procedure;

    -- Post-stall refill-bubble check for test_r0_no_refill_bubble_after_
    -- stall. With r1 contending for the same bank the whole time, r0's own
    -- inter-beat spacing is not a flat 1 cycle/beat even under correct RTL
    -- (bank contention interleaves the two channels' grants), so a
    -- per-beat spacing check (as used by 'check_full_rate' on an
    -- uncontested channel) does not apply here. Instead this checks r0's
    -- TOTAL transfer span against an exact cycle count: the whole pattern
    -- (no random-mode consumer on either channel) is fully deterministic,
    -- so that span is a single exact number under correct RTL, and a
    -- refill bubble after the stall can only inflate it.
    procedure check_no_refill_bubble(t_from : time; t_to : time; num_words : natural; stall_len : natural; msg : string) is
      variable span_cycles : natural;
    begin
      span_cycles := (t_to - t_from) / c_clk_period;
      info(
        msg & ": r0 span " & natural'image(span_cycles) & " cycles for " &
        natural'image(num_words) & " beats with a " & natural'image(stall_len) &
        "-cycle stall (bank 0 contended by r1 throughout)"
      );
      check_relation(
        span_cycles <= c_bubble_expected_span_cycles,
        msg & ": r0's transfer span exceeds the exact deterministic bound (" &
        natural'image(span_cycles) & " > " & natural'image(c_bubble_expected_span_cycles) &
        ") -- refill bubble after the stall?"
      );
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    do_reset;

    if run("test_r0_lossless_under_backpressure") then
      -- The primary regression test for the original bug (see this file's
      -- header comment). Phase 1: the directed drain-then-stall toggle
      -- pattern that triggers the T/T+1 double-issue deterministically.
      -- Phase 2: fully randomized backpressure (including multi-cycle
      -- stalls) on top of that, for broader coverage.
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 10);

      r0_mode <= c_mode_toggle;
      wait until rising_edge(clk);
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_bank_words / 2);
      wait until rising_edge(clk) and r0_done = '1';
      wait for c_settle;
      check_captured(r0_captured_q, c_bank_words / 2, 10, "r0 toggle-backpressure phase");

      r0_mode <= c_mode_random;
      r0_stall_pct <= 40;
      wait until rising_edge(clk);
      issue_read(r0_req_m2s, r0_req_s2m, 0, c_bank_words / 2, c_bank_words / 2);
      wait until rising_edge(clk) and r0_done = '1';
      wait for c_settle;
      check_captured(r0_captured_q, c_bank_words / 2, 10 + c_bank_words / 2, "r0 random-backpressure phase");

    elsif run("test_r1_lossless_under_backpressure") then
      -- Identical to test_r0_lossless_under_backpressure but on r1 -- r1
      -- had the same defect ('r1_can_issue' missing 'and not r1_capture').
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 20);

      r1_mode <= c_mode_toggle;
      wait until rising_edge(clk);
      issue_read(r1_req_m2s, r1_req_s2m, 0, 0, c_bank_words / 2);
      wait until rising_edge(clk) and r1_done = '1';
      wait for c_settle;
      check_captured(r1_captured_q, c_bank_words / 2, 20, "r1 toggle-backpressure phase");

      r1_mode <= c_mode_random;
      r1_stall_pct <= 40;
      wait until rising_edge(clk);
      issue_read(r1_req_m2s, r1_req_s2m, 0, c_bank_words / 2, c_bank_words / 2);
      wait until rising_edge(clk) and r1_done = '1';
      wait for c_settle;
      check_captured(r1_captured_q, c_bank_words / 2, 20 + c_bank_words / 2, "r1 random-backpressure phase");

    elsif run("test_concurrent_r0_r1_same_bank_no_loss_no_hol_blocking") then
      -- Both read channels active at once, targeting the SAME bank (so
      -- the read-side round-robin arbiter actually contends), each under
      -- its own independent backpressure. Proves the per-channel output
      -- register / capture tagging keeps the two streams fully isolated:
      -- no cross-channel beat loss and no head-of-line blocking (a slow
      -- channel must not stall the other).
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 100);

      r0_mode <= c_mode_toggle;
      r1_mode <= c_mode_random;
      r1_stall_pct <= 60;
      wait until rising_edge(clk);

      -- Both requests target bank 0, disjoint offset ranges within it, so
      -- both channels are simultaneously busy against the same bank's
      -- single physical read port for most of the run.
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_bank_words / 2);
      issue_read(r1_req_m2s, r1_req_s2m, 0, c_bank_words / 2, c_bank_words / 2);

      -- r0 (fast/toggle) and r1 (slow/random) finish on different, a
      -- priori unknown cycles -- wait for each 'done' pulse independently
      -- rather than requiring them to coincide.
      r0_seen := false;
      r1_seen := false;
      while not (r0_seen and r1_seen) loop
        wait until rising_edge(clk);
        if r0_done = '1' then
          r0_seen := true;
        end if;
        if r1_done = '1' then
          r1_seen := true;
        end if;
      end loop;
      wait for c_settle;

      check_captured(r0_captured_q, c_bank_words / 2, 100, "concurrent r0 (fast/toggle consumer)");
      check_captured(r1_captured_q, c_bank_words / 2, 100 + c_bank_words / 2, "concurrent r1 (slow/random consumer)");

    elsif run("test_throughput_r0_full_rate") then
      -- A lossless, non-backpressured read must sustain 1 beat/cycle.
      -- This is the acceptance test for the per-channel landing-register
      -- read path (see cnn_accel_tensor_mem.vhd's read_arbitrate header
      -- comment). An earlier, correct-but-throttling issue rule (one that
      -- forbade issuing on any cycle a beat was landing, with no skid
      -- register to catch it) serialized issue-then-capture and managed
      -- only 1 beat every 2 cycles, which this test would catch. Do not
      -- "fix" this test by loosening its bound; fix the RTL instead.
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 200);

      r0_mode <= c_mode_full_speed;
      wait until rising_edge(clk);

      -- 'issue_read' returns 'c_settle' after the edge on which the
      -- request was accepted, so back that offset out: 'check_full_rate'
      -- measures whole clock cycles from the acceptance edge.
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_bank_words);
      t_start := now - c_settle;
      wait until rising_edge(clk) and r0_done = '1';
      t_end := now;
      wait for c_settle;

      check_captured(r0_captured_q, c_bank_words, 200, "r0 throughput test data integrity");
      check_full_rate(t_start, t_end, c_bank_words, "test_throughput_r0_full_rate");

    elsif run("test_throughput_r1_full_rate") then
      -- Symmetric counterpart of test_throughput_r0_full_rate: r1 has its
      -- own landing register and must reach the same rate on its own.
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 300);

      r1_mode <= c_mode_full_speed;
      wait until rising_edge(clk);

      issue_read(r1_req_m2s, r1_req_s2m, 0, 0, c_bank_words);
      t_start := now - c_settle;
      wait until rising_edge(clk) and r1_done = '1';
      t_end := now;
      wait for c_settle;

      check_captured(r1_captured_q, c_bank_words, 300, "r1 throughput test data integrity");
      check_full_rate(t_start, t_end, c_bank_words, "test_throughput_r1_full_rate");

    elsif run("test_throughput_both_channels_full_rate_different_banks") then
      -- Both read channels streaming at once from *different* banks, so
      -- each has a bank read port to itself and both must sustain the
      -- full 1 beat/cycle simultaneously. (Aimed at the same bank they
      -- would necessarily share one physical read port and average half
      -- rate each; that case is covered for correctness, not rate, by
      -- test_concurrent_r0_r1_same_bank_no_loss_no_hol_blocking.)
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 400);
      write_words(w1_req_m2s, w1_req_s2m, s_w1_m2s, s_w1_s2m, 1, 0, c_bank_words, 500);

      r0_mode <= c_mode_full_speed;
      r1_mode <= c_mode_full_speed;
      wait until rising_edge(clk);

      -- Requests are accepted one cycle apart (this process can only
      -- issue one per cycle), so each channel is timed from its own
      -- acceptance edge rather than from a shared start.
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_bank_words);
      t_start := now - c_settle;
      issue_read(r1_req_m2s, r1_req_s2m, 1, 0, c_bank_words);
      t_start_r1 := now - c_settle;

      r0_seen := false;
      r1_seen := false;
      while not (r0_seen and r1_seen) loop
        wait until rising_edge(clk);
        if r0_done = '1' and not r0_seen then
          r0_seen := true;
          t_end := now;
        end if;
        if r1_done = '1' and not r1_seen then
          r1_seen := true;
          t_end_r1 := now;
        end if;
      end loop;
      wait for c_settle;

      check_captured(r0_captured_q, c_bank_words, 400, "concurrent full-rate r0 data integrity");
      check_captured(r1_captured_q, c_bank_words, 500, "concurrent full-rate r1 data integrity");
      check_full_rate(t_start, t_end, c_bank_words, "concurrent full-rate r0 (bank 0)");
      check_full_rate(t_start_r1, t_end_r1, c_bank_words, "concurrent full-rate r1 (bank 1)");

    elsif run("test_r0_no_refill_bubble_after_stall") then
      -- Targets the '(not r0_skid_valid and not r0_capture)' disjunct in
      -- 'r0_can_issue' specifically, as opposed to the skid register it
      -- guards (see cnn_accel_tensor_mem.vhd's read_arbitrate header
      -- comment and the comment on 'r0_can_issue'/'r1_can_issue' for the
      -- division of responsibility). The skid register plus the plain
      -- 'not r0_out_valid or m_r0_s2m.ready' term is already enough for
      -- losslessness -- test_r0_lossless_under_backpressure and the
      -- throughput tests would all still pass without this disjunct.
      --
      -- Why r1 is also active here, contending for the SAME bank: with r0
      -- running alone, 'not r0_out_valid' already lets it launch two reads
      -- back to back during a request's initial pipeline ramp (before the
      -- output register has anything in it), so a beat is always
      -- coincidentally already in flight by the time the output register
      -- first fills -- and that coincidence, not the disjunct under test,
      -- is what fills the skid register during a later stall. That makes
      -- the disjunct's effect unobservable with r0 alone: it was tried
      -- first, and it does not distinguish (every existing r0-only test,
      -- including this one in an earlier form, passes with the disjunct
      -- removed). Running r1 at full speed against the same bank the whole
      -- time forces the read-side round-robin arbiter to interleave the
      -- two channels' grants, which breaks that ramp-up coincidence for
      -- r0: r0's reads no longer arrive in an uninterrupted back-to-back
      -- pair, so whether it can claim a *second* bank grant while stalled
      -- and the output register is occupied genuinely depends on this
      -- disjunct.
      --
      -- Directed pattern: r1 streams a disjoint offset range of the same
      -- bank at full speed for the entire test (so bank 0 is contended for
      -- every cycle either channel wants it). r0 streams its own disjoint
      -- range, reaches steady state, then its consumer holds 'ready' low
      -- for 'c_stall_len' cycles mid-burst before holding it high for the
      -- rest of the transfer. The whole pattern is deterministic (no
      -- random-mode consumer involved on either channel), so r0's total
      -- transfer span is an exact cycle count, not a statistical average --
      -- checked with the same exact-allowance style as 'check_full_rate'.
      write_words(w0_req_m2s, w0_req_s2m, s_w0_m2s, s_w0_s2m, 0, 0, c_bank_words, 600);

      r0_mode <= c_mode_full_speed;
      r1_mode <= c_mode_full_speed;
      r0_pause <= '0';
      wait until rising_edge(clk);

      -- Start r1's contention first so bank 0 is already contended from
      -- r0's very first cycle onward.
      issue_read(r1_req_m2s, r1_req_s2m, 0, c_bubble_r0_words, c_bank_words - c_bubble_r0_words);
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_bubble_r0_words);
      t_start := now - c_settle;

      -- Run long enough (well past 'c_read_pipeline_cycles') to reach
      -- steady state before pausing, and leave plenty of beats after the
      -- stall for the post-stall spacing to be checked.
      for i in 1 to c_stall_start_cycles loop
        wait until rising_edge(clk);
      end loop;
      r0_pause <= '1';
      for i in 1 to c_stall_len loop
        wait until rising_edge(clk);
      end loop;
      r0_pause <= '0';

      wait until rising_edge(clk) and r0_done = '1';
      t_end := now;
      wait until rising_edge(clk) and r1_done = '1';
      wait for c_settle;

      check_captured(r0_captured_q, c_bubble_r0_words, 600, "r0 no-refill-bubble-after-stall data integrity");
      check_captured(
        r1_captured_q, c_bank_words - c_bubble_r0_words, 600 + c_bubble_r0_words,
        "r0 no-refill-bubble-after-stall: r1 contention data integrity"
      );
      check_no_refill_bubble(t_start, t_end, c_bubble_r0_words, c_stall_len, "test_r0_no_refill_bubble_after_stall");

    elsif run("test_write_w0_w1_same_bank_round_robin") then
      -- Write-side counterpart of
      -- test_concurrent_r0_r1_same_bank_no_loss_no_hol_blocking: every
      -- other write-side test only ever drives 'w1' sequentially (a
      -- different bank, or after 'w0' has already finished), so the write
      -- round-robin arbiter (cnn_accel_tensor_mem.vhd's write_arbitrate
      -- process) is never actually exercised under contention. This test
      -- drives both write channels into the SAME bank concurrently, each
      -- holding 'valid' high for its entire transfer, so every cycle while
      -- both still have data left is a contended cycle. That makes the
      -- arbiter's "who won last time" alternation checkable cycle by
      -- cycle, not just as an aggregate throughput number -- proving
      -- neither channel can be starved, not merely that neither happened
      -- to be starved this run.
      issue_write(w0_req_m2s, w0_req_s2m, 0, 0, c_write_contend_words);
      issue_write(w1_req_m2s, w1_req_s2m, 0, c_write_contend_words, c_write_contend_words);

      wr_w0_left := c_write_contend_words;
      wr_w1_left := c_write_contend_words;
      wr_w0_idx := 0;
      wr_w1_idx := 0;
      wr_have_prev := false;

      s_w0_m2s.data <= (others => '0');
      s_w0_m2s.data(c_data_width - 1 downto 0) <= word_value(700, 0);
      s_w0_m2s.valid <= '1';
      s_w1_m2s.data <= (others => '0');
      s_w1_m2s.data(c_data_width - 1 downto 0) <= word_value(800, 0);
      s_w1_m2s.valid <= '1';

      -- Loop until both channels' 'done' pulses have been observed --
      -- checked every cycle inside this same loop, not in a separate loop
      -- afterward, since a channel's one-cycle 'done' pulse can fall on the
      -- very last iteration where its beat counter reaches zero (the two
      -- channels do not finish on the same cycle, since the arbiter grants
      -- only one of them per contended cycle) and would otherwise be
      -- missed between the two loops.
      wr_w0_done_seen := false;
      wr_w1_done_seen := false;
      while not (wr_w0_done_seen and wr_w1_done_seen) loop
        wr_both_wanted := wr_w0_left > 0 and wr_w1_left > 0;
        wait until rising_edge(clk);

        wr_w0_won := s_w0_s2m.ready = '1' and wr_w0_left > 0;
        wr_w1_won := s_w1_s2m.ready = '1' and wr_w1_left > 0;

        check_false(
          wr_w0_won and wr_w1_won,
          "test_write_w0_w1_same_bank_round_robin: both channels granted the same bank on the same cycle"
        );

        if wr_both_wanted then
          if wr_have_prev then
            check_true(
              (wr_w0_won and wr_prev_grant = 1) or (wr_w1_won and wr_prev_grant = 0),
              "test_write_w0_w1_same_bank_round_robin: arbiter did not alternate on sustained same-bank contention " &
              "(one channel granted twice in a row -- possible starvation)"
            );
          end if;
          if wr_w0_won then
            wr_prev_grant := 0;
            wr_have_prev := true;
          elsif wr_w1_won then
            wr_prev_grant := 1;
            wr_have_prev := true;
          end if;
        end if;

        if wr_w0_won then
          wr_w0_left := wr_w0_left - 1;
          wr_w0_idx := wr_w0_idx + 1;
          if wr_w0_left = 0 then
            s_w0_m2s.valid <= '0';
          else
            s_w0_m2s.data(c_data_width - 1 downto 0) <= word_value(700, wr_w0_idx);
          end if;
        end if;

        if wr_w1_won then
          wr_w1_left := wr_w1_left - 1;
          wr_w1_idx := wr_w1_idx + 1;
          if wr_w1_left = 0 then
            s_w1_m2s.valid <= '0';
          else
            s_w1_m2s.data(c_data_width - 1 downto 0) <= word_value(800, wr_w1_idx);
          end if;
        end if;

        if w0_done = '1' then
          wr_w0_done_seen := true;
        end if;
        if w1_done = '1' then
          wr_w1_done_seen := true;
        end if;
      end loop;
      wait for c_settle;

      -- Read each channel's range back independently and check the data
      -- landed correctly -- proof that concurrent same-bank contention
      -- corrupted neither channel's beats.
      r0_mode <= c_mode_full_speed;
      wait until rising_edge(clk);
      issue_read(r0_req_m2s, r0_req_s2m, 0, 0, c_write_contend_words);
      wait until rising_edge(clk) and r0_done = '1';
      wait for c_settle;
      check_captured(r0_captured_q, c_write_contend_words, 700, "w0/w1 same-bank contention: w0 data readback");

      wait until rising_edge(clk);
      issue_read(r0_req_m2s, r0_req_s2m, 0, c_write_contend_words, c_write_contend_words);
      wait until rising_edge(clk) and r0_done = '1';
      wait for c_settle;
      check_captured(r0_captured_q, c_write_contend_words, 800, "w0/w1 same-bank contention: w1 data readback");
    end if;

    test_runner_cleanup(runner);
    wait;
  end process;

  test_runner_watchdog(runner, 2 ms);

end architecture tb;
