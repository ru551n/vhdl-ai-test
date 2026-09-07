library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- Generic AXI4 read master. See modules/cnn_accel/doc/cnn_accel_axi_read_dma_req.md
-- and modules/cnn_accel/doc/cnn_accel_axi_read_dma_proposal.md for the full
-- rationale behind every design decision below.
--
-- Accepts one 'dma_req_t' (byte 'addr'/'length') at a time, splits it into
-- one or more 'AR' bursts that respect the 4 KiB burst-boundary rule and the
-- 256-beat 'ARLEN' limit, issues them back-to-back through the reused
-- 'axi.axi_read_pipeline'/'axi.axi_read_throttle' (throttled against a real
-- 'axi.axi_r_fifo' holding the outstanding 'R' beats, per that entity's own
-- documented usage pattern), and republishes every accepted 'R' beat as an
-- internal AXI4-Stream through an 'axi_stream_fifo' elasticity stage.
-- 'dma_done' pulses once the whole request has been streamed out to the
-- consumer (i.e. once the beat carrying 'last' has been accepted at
-- 'm_stream_s2m'); 'resp_error' pulses on the same cycle if any 'RRESP' in
-- the request was not 'OKAY' (the burst is still fully drained per
-- shared/Axi4.md rule 15). 'ARID'/'RID' is a hardwired '0' (proposal doc
-- section 3.1) -- not a generic -- since 'axi.axi_simple_read_crossbar'
-- (the documented downstream consumer of 'm_axi_ar'/'m_axi_r') port-locks
-- for a whole burst, so no two instances' traffic is ever interleaved
-- there regardless of the 'ARID' value each one drives.
--
-- Word-aligned assumption (proposal doc section 3.4): 'req_m2s.req.addr'/
-- '.length' are both multiples of 'g_axi_data_width/8' -- honored by every
-- caller ('cnn_accel_sequencer', 'cnn_accel_layer_ctrl'). 'length = 0' is
-- accepted as a degenerate, immediate-'dma_done' case (no 'AR' issued at
-- all).
--
-- Reset (proposal doc section 4): synchronous active-high 'reset'
-- ('reset_internal'). On 'reset = ''1''' the top request FSM returns to
-- 'IDLE' ('req_s2m.ready' high again the next cycle) and 'err_pending_q'
-- clears; the reused hdl-modules submodules ('axi_read_pipeline',
-- 'axi_read_throttle', 'axi_r_fifo', 'axi_stream_fifo') have no 'reset'
-- port of their own (hdl-modules' resetless-by-default convention), so any
-- beat already in flight inside them at the moment of 'reset' is *not*
-- discarded there -- it will still surface at 'axi_r_fifo's consumer side
-- at some later, real-bus-timing-dependent point, same as any 'AR' this
-- entity had already handed to those skid buffers (whose one-cycle,
-- unconditional-capture behavior means even an abort on the very next
-- cycle cannot prevent that hand-off; see each reused entity's own doc).
-- This entity accounts for exactly that: on 'reset', 'stale_beats_q' is
-- incremented by however many beats the just-abandoned request had
-- already gotten accepted into the pipe ('beats_issued_q') but not yet
-- counted as consumed ('r_beat_count_q'); a subsequently accepted request
-- is held in the internal 'DRAIN_STALE' state -- 'req_s2m.ready' already
-- low, no 'AR' issued yet, nothing forwarded to 'm_stream_m2s' -- for as
-- long as 'stale_beats_q' is nonzero, patiently popping and discarding
-- exactly that many leftover beats off 'axi_r_fifo' before ever issuing
-- its own first 'AR' or counting anything towards its own 'total_beats_q'.
-- This works regardless of when the abort happens relative to the
-- abandoned request's 'AR'/'R' progress, and regardless of how many
-- aborts stack up back-to-back ('stale_beats_q' is deliberately not
-- cleared by 'reset' itself, only ever incremented by it or decremented by
-- 'DRAIN_STALE', so it correctly accumulates and survives across
-- consecutive aborts). Exercised by the testbench's
-- 'test_reset_mid_transfer' case.
entity cnn_accel_axi_read_dma is
  generic (
    g_axi_addr_width : positive;
    g_axi_data_width : positive;
    g_axi_id_width : natural
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    req_m2s : in dma_req_m2s_t;
    req_s2m : out dma_req_s2m_t;
    dma_done : out std_ulogic := '0';
    resp_error : out std_ulogic := '0';
    --# {{}}
    m_axi_ar_m2s : out axi_m2s_a_t := axi_m2s_a_init;
    m_axi_ar_s2m : in axi_s2m_a_t;
    -- Direction fixed vs. the (defective) requirement port table -- 'RREADY'
    -- is this master's output, 'RVALID'/'RDATA'/'RRESP'/'RLAST' are inputs,
    -- mirroring the 'AR' row exactly. See proposal doc section 3.2.
    m_axi_r_m2s : out axi_m2s_r_t := axi_m2s_r_init;
    m_axi_r_s2m : in axi_s2m_r_t;
    --# {{}}
    m_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_stream_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_axi_read_dma;

architecture a of cnn_accel_axi_read_dma is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_bytes_per_beat : positive := g_axi_data_width / 8;
  constant c_max_burst_bytes : positive := axi_max_burst_length_beats * c_bytes_per_beat;

  -- Must be at least 2x the max burst length, not 1x: 'axi_read_throttle's
  -- own 'block_address_transactions' logic blocks issuing a new AR whenever
  -- the *current* burst's length is >= the FIFO's empty-and-not-yet-
  -- negotiated space, so a FIFO sized to exactly one max-length burst
  -- deadlocks the moment a full 256-beat burst is negotiated but not yet
  -- drained (its own empty space drops to exactly 0, blocking that same
  -- burst's own AR forever). Matches the ratio used by
  -- 'axi_read_throttle's own reference testbench (proposal doc section
  -- 3.6, corrected). Found via 'test_multi_burst_over_256_beats' deadlock.
  constant c_r_fifo_depth : positive := 2 * axi_max_burst_length_beats;
  -- Local constant, not a generic (proposal doc section 3.7).
  constant c_stream_fifo_depth : positive := 16;

  ------------------------------------------------------------------------
  -- Top request FSM + AR-issue/R-consume registers (proposal doc section 6).
  ------------------------------------------------------------------------

  -- 's_drain_stale' (proposal doc section 4 addendum below): a request is
  -- routed through this state, ahead of 's_active', whenever a prior abort
  -- (mid-transfer 'reset') left beats still owed from an AR that the
  -- resetless 'axi_read_throttle'/'axi_read_pipeline' skid buffers had
  -- already latched -- see 'stale_beats_q' below.
  type state_t is (s_idle, s_drain_stale, s_active, s_wait_drain, s_done_pulse);
  signal state_q : state_t := s_idle;

  signal addr_q : unsigned(31 downto 0) := (others => '0');
  signal bytes_remaining_q : unsigned(31 downto 0) := (others => '0');
  signal total_beats_q : unsigned(31 downto 0) := (others => '0');
  signal r_beat_count_q : unsigned(31 downto 0) := (others => '0');
  signal err_pending_q : std_ulogic := '0';

  -- Cumulative beats across every AR accepted so far for the *current*
  -- request (i.e. every 'ar_accepted_i' since the request was latched in
  -- 's_idle') -- used only to compute 'stale_beats_q' on an abort; not
  -- otherwise part of the documented interface.
  signal beats_issued_q : unsigned(31 downto 0) := (others => '0');

  -- Number of beats not yet counted in 'r_beat_count_q' that are still
  -- owed by an *abandoned* request's already-accepted AR(s) -- i.e. beats
  -- that the resetless 'axi_read_throttle'/'axi_read_pipeline' skid
  -- buffers (see their own doc: a one-cycle-unconditional-capture skid
  -- buffer with no 'reset' port) may have already latched, or the real R
  -- channel may still be in the process of delivering, at the moment this
  -- entity's own 'reset' aborted that request. Deliberately *not* cleared
  -- by 'reset' itself (unlike every other register above) -- it is only
  -- ever incremented (by 'reset', capturing what the just-aborted request
  -- still owed) or decremented (by 's_drain_stale' below, as each such
  -- beat is actually popped off 'axi_r_fifo' and discarded) so that it
  -- correctly survives back-to-back aborts and is never lost. A freshly
  -- accepted request is held in 's_drain_stale' -- not issuing its own AR
  -- yet -- for as long as this is nonzero, so that this leftover data from
  -- the abandoned request can never be mistaken for the new request's
  -- leading beats (proposal doc section 4's documented gap, now closed).
  signal stale_beats_q : unsigned(31 downto 0) := (others => '0');

  -- This request's next AR burst, computed combinationally from 'addr_q'/
  -- 'bytes_remaining_q' (proposal doc section 6.2) -- stable between
  -- acceptances since both registers only change on an accepted AR.
  signal burst_bytes_c : unsigned(31 downto 0) := (others => '0');
  signal burst_beats_c : positive range 1 to axi_max_burst_length_beats := 1;

  signal ar_issue_active_i : std_ulogic;
  signal ar_accepted_i : std_ulogic;

  signal r_pop_i : std_ulogic;
  -- 'r_pop_i' qualified by which state it happened in: '_active' feeds the
  -- real request's bookkeeping/stream output, '_drain' feeds 'stale_beats_q'.
  signal r_pop_active_i : std_ulogic;
  signal r_pop_drain_i : std_ulogic;
  signal last_beat_i : std_ulogic;

  ------------------------------------------------------------------------
  -- Internal bundled AXI4 read bus (AR+R together), FSM -> throttle ->
  -- pipeline -> split to the four 'm_axi_*' ports at the entity boundary
  -- (proposal doc section 3.3/section 5).
  ------------------------------------------------------------------------

  signal throttle_input_m2s : axi_read_m2s_t := axi_read_m2s_init;
  signal throttle_input_s2m : axi_read_s2m_t := axi_read_s2m_init;

  signal throttled_m2s : axi_read_m2s_t := axi_read_m2s_init;
  signal throttled_s2m : axi_read_s2m_t := axi_read_s2m_init;

  signal pipeline_right_m2s : axi_read_m2s_t := axi_read_m2s_init;
  signal pipeline_right_s2m : axi_read_s2m_t := axi_read_s2m_init;

  ------------------------------------------------------------------------
  -- 'axi_r_fifo': "input" = consumer/pop side (read by R-consume below),
  -- "output" = upstream/write side (fed from 'throttle_input_s2m.r', i.e.
  -- the real R data after throttle+pipeline). Feeds
  -- 'axi_read_throttle.data_fifo_level' (proposal doc section 3.6).
  ------------------------------------------------------------------------

  signal r_fifo_input_m2s : axi_m2s_r_t := axi_m2s_r_init;
  signal r_fifo_input_s2m : axi_s2m_r_t := axi_s2m_r_init;
  signal r_fifo_output_m2s : axi_m2s_r_t := axi_m2s_r_init;
  signal r_fifo_level : natural range 0 to c_r_fifo_depth := 0;

  ------------------------------------------------------------------------
  -- Output elasticity stage (proposal doc section 3.7); 'output_m2s'/
  -- 'output_s2m' are wired straight to the entity's 'm_stream_m2s'/'s2m'
  -- ports at the instantiation below.
  ------------------------------------------------------------------------

  signal stream_fifo_input_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal stream_fifo_input_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

begin

  ------------------------------------------------------------------------
  -- Elaboration-time bus-width bound (decision S6 + decision D1, 2026-09-07).
  --
  -- This entity's word-aligned assumption ('req.addr'/'.length' both
  -- multiples of 'g_axi_data_width/8') fails silently, not loudly: an
  -- under-length final chunk simply never completes and 'dma_done' never
  -- fires, hanging the layer.
  --
  -- Of this entity's three instances in the IP, the instruction fetch
  -- (64-byte descriptors) and the weight/bias fetch
  -- ('K^2 * n_in_tiles * 64' and 'g_pe_rows * 4' bytes) satisfy the
  -- assumption at any AXI-legal width. The IFMAP instance does not: under
  -- decision S6 it reads whole channel-tiled planes
  -- ('in_width * in_height * T' bytes at 'ifmap_addr + ct * plane_len'), so
  -- its requests are T-byte granular and nothing more. The bound below is
  -- therefore what makes the ifmap path aligned unconditionally, for every
  -- layer geometry, with no runtime check.
  --
  -- It is asserted for all three instances rather than only the ifmap one:
  -- all three take 'g_axi_data_width' from a single top-level generic, so a
  -- per-instance distinction could not be configured independently anyway,
  -- and over-constraining here fails loudly at elaboration -- the safe
  -- direction for a contract whose violation is otherwise a silent hang.
  ------------------------------------------------------------------------

  assert g_axi_data_width <= cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_max_axi_data_width
    report "cnn_accel_axi_read_dma: g_axi_data_width (" & positive'image(g_axi_data_width) &
      ") exceeds the S6 activation-plane bound (" &
      integer'image(cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_max_axi_data_width) &
      " bits = " &
      integer'image(cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_activation_plane_channels) &
      " bytes/beat). The ifmap instance reads whole S6 activation planes, " &
      "which are only T-byte granular, so a wider bus makes addr/length " &
      "alignment depend on runtime layer geometry, and an unaligned request " &
      "hangs (dma_done never fires) instead of erroring. See " &
      "doc/cnn_accel_arch.md 'Off-chip activation layout (decision S6)'."
    severity failure;

  ------------------------------------------------------------------------
  -- AR-issue sub-logic (proposal doc section 6.2): burst split against the
  -- 4 KiB boundary and the 256-beat ARLEN limit, computed combinationally
  -- from the currently latched 'addr_q'/'bytes_remaining_q'.
  ------------------------------------------------------------------------

  burst_calc : process(all)
    variable v_bytes_to_4k : unsigned(31 downto 0);
    variable v_burst_bytes : unsigned(31 downto 0);
  begin
    if addr_q(11 downto 0) = 0 then
      v_bytes_to_4k := to_unsigned(4096, 32);
    else
      v_bytes_to_4k := to_unsigned(4096, 32) - resize(addr_q(11 downto 0), 32);
    end if;

    v_burst_bytes := bytes_remaining_q;
    if v_bytes_to_4k < v_burst_bytes then
      v_burst_bytes := v_bytes_to_4k;
    end if;
    if to_unsigned(c_max_burst_bytes, 32) < v_burst_bytes then
      v_burst_bytes := to_unsigned(c_max_burst_bytes, 32);
    end if;

    burst_bytes_c <= v_burst_bytes;

    if v_burst_bytes = 0 then
      -- No burst pending ('bytes_remaining_q = 0') -- dummy, safe value;
      -- never actually issued since 'ar_issue_active_i' is low whenever
      -- 'bytes_remaining_q = 0'.
      burst_beats_c <= 1;
    else
      burst_beats_c <= to_integer(v_burst_bytes) / c_bytes_per_beat;
    end if;
  end process;

  ar_issue_active_i <= '1' when (state_q = s_active and bytes_remaining_q /= 0) else '0';
  ar_accepted_i <= '1' when (ar_issue_active_i = '1' and throttle_input_s2m.ar.ready = '1') else '0';

  throttle_input_m2s.ar.valid <= ar_issue_active_i;
  throttle_input_m2s.ar.id <= (others => '0');
  throttle_input_m2s.ar.addr <= resize(addr_q, throttle_input_m2s.ar.addr'length);
  throttle_input_m2s.ar.len <= to_len(burst_beats_c);
  throttle_input_m2s.ar.size <= to_size(g_axi_data_width);
  throttle_input_m2s.ar.burst <= axi_a_burst_incr;


  ------------------------------------------------------------------------
  -- R-consume sub-logic (proposal doc section 6.3): pops 'axi_r_fifo's
  -- consumer side whenever the output 'axi_stream_fifo' has room, tags
  -- 'last' on the global beat counter's final beat, latches
  -- 'err_pending_q' on a non-OKAY 'RRESP' without stopping the pop (rule
  -- 15 -- the burst still fully drains).
  ------------------------------------------------------------------------

  throttle_input_m2s.r <= r_fifo_output_m2s;

  -- Gated to 'ACTIVE'/'DRAIN_STALE' only (proposal doc section 6.3/section
  -- 4): prevents popping stale beats left over in 'axi_r_fifo' by an
  -- abandoned request once back in 'IDLE'/'WAIT_DRAIN'/'DONE_PULSE'.
  -- 'DRAIN_STALE' pops unconditionally (no stream backpressure to honor,
  -- since this data is discarded, not forwarded); 'ACTIVE' pops only when
  -- the output stream FIFO has room, same as before.
  r_fifo_input_m2s.ready <=
    stream_fifo_input_s2m.ready when state_q = s_active else
    '1' when state_q = s_drain_stale else
    '0';

  r_pop_i <= '1' when (r_fifo_input_s2m.valid = '1' and r_fifo_input_m2s.ready = '1') else '0';
  r_pop_active_i <= '1' when (r_pop_i = '1' and state_q = s_active) else '0';
  r_pop_drain_i <= '1' when (r_pop_i = '1' and state_q = s_drain_stale) else '0';
  last_beat_i <= '1' when (r_beat_count_q + 1 = total_beats_q) else '0';

  stream_fifo_input_m2s.valid <= r_pop_active_i;
  stream_fifo_input_m2s.last <= last_beat_i;
  stream_fifo_input_m2s.user <= (others => '0');
  stream_fifo_input_m2s.data <= std_ulogic_vector(
    resize(unsigned(r_fifo_input_s2m.data(g_axi_data_width - 1 downto 0)), stream_fifo_input_m2s.data'length)
  );


  ------------------------------------------------------------------------
  -- Top-level handshake/pulse outputs (proposal doc section 3.8/section
  -- 6.1): all combinational, in step with 'state_q'.
  ------------------------------------------------------------------------

  req_s2m.ready <= '1' when state_q = s_idle else '0';
  dma_done <= '1' when state_q = s_done_pulse else '0';
  resp_error <= '1' when (state_q = s_done_pulse and err_pending_q = '1') else '0';


  ------------------------------------------------------------------------
  -- Top request FSM (proposal doc section 6.1).
  ------------------------------------------------------------------------

  main : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        state_q <= s_idle;
        addr_q <= (others => '0');
        bytes_remaining_q <= (others => '0');
        total_beats_q <= (others => '0');
        r_beat_count_q <= (others => '0');
        err_pending_q <= '0';
        beats_issued_q <= (others => '0');

        -- Whatever this (now-abandoned) request had already gotten
        -- accepted into the pipe ('beats_issued_q') but not yet counted as
        -- consumed ('r_beat_count_q') is still owed by 'axi_r_fifo' at some
        -- point in the future (see 'stale_beats_q's declaration comment).
        -- 'stale_beats_q' itself is deliberately *not* reset above so a
        -- second abort before the first is fully drained still accumulates
        -- correctly.
        if beats_issued_q > r_beat_count_q then
          stale_beats_q <= stale_beats_q + (beats_issued_q - r_beat_count_q);
        end if;
      else
        if ar_accepted_i = '1' then
          addr_q <= addr_q + burst_bytes_c;
          bytes_remaining_q <= bytes_remaining_q - burst_bytes_c;
          beats_issued_q <= beats_issued_q + burst_beats_c;
        end if;

        if r_pop_active_i = '1' then
          r_beat_count_q <= r_beat_count_q + 1;

          if r_fifo_input_s2m.resp /= axi_resp_okay then
            err_pending_q <= '1';
          end if;
        end if;

        if r_pop_drain_i = '1' then
          stale_beats_q <= stale_beats_q - 1;
        end if;

        case state_q is
          when s_idle =>
            if req_m2s.valid = '1' then
              addr_q <= req_m2s.req.addr;
              bytes_remaining_q <= req_m2s.req.length;
              total_beats_q <= req_m2s.req.length / to_unsigned(c_bytes_per_beat, 32);
              r_beat_count_q <= (others => '0');
              err_pending_q <= '0';
              beats_issued_q <= (others => '0');

              if req_m2s.req.length = 0 then
                -- Degenerate case: no 'AR' issued at all (proposal doc
                -- section 3.8). Any 'stale_beats_q' owed from an earlier
                -- abort is left untouched -- drained ahead of the next
                -- non-zero-length request instead (see 'stale_beats_q's
                -- declaration comment).
                state_q <= s_done_pulse;
              elsif stale_beats_q /= 0 then
                -- Must not start issuing this request's own AR (nor count
                -- any 'axi_r_fifo' data as belonging to it) until every
                -- beat still owed by a previously abandoned request has
                -- been popped off and discarded -- otherwise that leftover
                -- data would be mistaken for this request's leading beats.
                state_q <= s_drain_stale;
              else
                state_q <= s_active;
              end if;
            end if;

          when s_drain_stale =>
            if r_pop_drain_i = '1' and stale_beats_q = 1 then
              state_q <= s_active;
            end if;

          when s_active =>
            if r_pop_active_i = '1' and r_beat_count_q + 1 = total_beats_q then
              state_q <= s_wait_drain;
            end if;

          when s_wait_drain =>
            -- Wait until the beat carrying 'last' leaves the output
            -- elasticity FIFO at the true external boundary.
            if m_stream_m2s.valid = '1' and m_stream_m2s.last = '1' and m_stream_s2m.ready = '1' then
              state_q <= s_done_pulse;
            end if;

          when s_done_pulse =>
            state_q <= s_idle;
        end case;
      end if;
    end if;
  end process;


  ------------------------------------------------------------------------
  -- Reused hdl-modules building blocks (shared/ReusableRTL.md).
  ------------------------------------------------------------------------

  axi_read_throttle_inst : entity axi.axi_read_throttle
    generic map (
      data_fifo_depth => c_r_fifo_depth,
      max_burst_length_beats => axi_max_burst_length_beats,
      id_width => g_axi_id_width,
      addr_width => g_axi_addr_width,
      full_ar_throughput => true
    )
    port map (
      clk => clk,
      --
      data_fifo_level => r_fifo_level,
      --
      input_m2s => throttle_input_m2s,
      input_s2m => throttle_input_s2m,
      --
      throttled_m2s => throttled_m2s,
      throttled_s2m => throttled_s2m
    );

  axi_read_pipeline_inst : entity axi.axi_read_pipeline
    generic map (
      addr_width => g_axi_addr_width,
      id_width => g_axi_id_width,
      data_width => g_axi_data_width
    )
    port map (
      clk => clk,
      --
      left_m2s => throttled_m2s,
      left_s2m => throttled_s2m,
      --
      right_m2s => pipeline_right_m2s,
      right_s2m => pipeline_right_s2m
    );

  -- Boundary-only split into the four 'm_axi_*' ports (proposal doc
  -- section 3.3) -- no behavior added.
  m_axi_ar_m2s <= pipeline_right_m2s.ar;
  pipeline_right_s2m.ar <= m_axi_ar_s2m;
  m_axi_r_m2s <= pipeline_right_m2s.r;
  pipeline_right_s2m.r <= m_axi_r_s2m;

  axi_r_fifo_inst : entity axi.axi_r_fifo
    generic map (
      asynchronous => false,
      id_width => g_axi_id_width,
      data_width => g_axi_data_width,
      depth => c_r_fifo_depth
    )
    port map (
      clk => clk,
      --
      input_m2s => r_fifo_input_m2s,
      input_s2m => r_fifo_input_s2m,
      --
      output_m2s => r_fifo_output_m2s,
      output_s2m => throttle_input_s2m.r,
      output_level => r_fifo_level
    );

  axi_stream_fifo_inst : entity axi_stream.axi_stream_fifo
    generic map (
      data_width => g_axi_data_width,
      user_width => 0,
      asynchronous => false,
      depth => c_stream_fifo_depth
    )
    port map (
      clk => clk,
      --
      input_m2s => stream_fifo_input_m2s,
      input_s2m => stream_fifo_input_s2m,
      --
      output_m2s => m_stream_m2s,
      output_s2m => m_stream_s2m
    );

end architecture a;
