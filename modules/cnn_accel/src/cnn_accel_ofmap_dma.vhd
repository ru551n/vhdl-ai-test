library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.log2;

library dma_axi_write_simple;
use dma_axi_write_simple.dma_axi_write_simple_register_record_pkg.all;

-- Thin wrapper around hdl-modules 'dma_axi_write_simple.dma_axi_write_simple'
-- (the plain, non-AXI-Lite entity, reused unmodified) that adapts its native
-- ring-buffer register interface to this IP's one-shot 'dma_req_m2s_t'/
-- 's2m_t' (addr+length) control plane. See
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_req.md and
-- modules/cnn_accel/doc/cnn_accel_ofmap_dma_proposal.md for the full
-- rationale behind every design decision below.
--
-- Per request: 'buffer_start_address <= addr', 'buffer_end_address <= addr +
-- length + one segment's worth of padding', 'buffer_read_address <=
-- buffer_start_address' (kept equal to the start address at all times, to
-- satisfy 'ring_buffer_write_simple's own "initial read address should be
-- start address" simulation assertion; the one-segment padding on
-- 'buffer_end_address' is what still lets exactly this request's segment
-- count through before the ring buffer's own "never more than size-1
-- segments outstanding" rule self-blocks it -- see the proposal doc's
-- Implementation Notes). 'config.enable' is pulsed combinationally for
-- exactly the duration of this request (belt-and-suspenders self-limited
-- by 'aw_issued_q' too). 'packet_length_beats => 1' and
-- 'stream_data_width => axi_data_width => g_axi_data_width' land the wrapped
-- core on its single-AXI-beat-packet optimized implementation and avoid its
-- internal (reset-less) 'width_conversion' block entirely -- see proposal
-- doc section 3.3/3.4. Completion/error tracking taps 'm_axi_b_*' directly
-- (proposal doc section 3.5), since one packet = one AXI beat here.
--
-- The wrapped core has no reset port at all (hdl-modules' resetless-by-
-- default convention). This wrapper never asks it to abandon an in-flight
-- 'AW'/'W' transaction (not a wrapped-module limitation -- a basic AXI4
-- rule); "abort" means stop issuing new bursts and drain already-issued
-- ones. 'outstanding_q' tracks physical AW-accepted-but-B-not-yet-seen
-- transactions and deliberately survives 'reset' to stay physically
-- accurate; every other piece of state is cleared by 'reset'. See proposal
-- doc section 4 for the full reset/abort design.
entity cnn_accel_ofmap_dma is
  generic (
    g_axi_addr_width : positive;
    g_axi_data_width : positive
  );
  port (
    clk : in std_ulogic;
    reset : in std_ulogic := '0';
    --# {{}}
    req_m2s : in dma_req_m2s_t;
    req_s2m : out dma_req_s2m_t;
    dma_done : out std_ulogic := '0';
    resp_error : out std_ulogic := '0';
    --# {{}}
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;
    --# {{}}
    m_axi_aw_m2s : out axi_m2s_a_t := axi_m2s_a_init;
    m_axi_aw_s2m : in axi_s2m_a_t;
    m_axi_w_m2s : out axi_m2s_w_t := axi_m2s_w_init;
    m_axi_w_s2m : in axi_s2m_w_t;
    m_axi_b_m2s : out axi_m2s_b_t := axi_m2s_b_init;
    m_axi_b_s2m : in axi_s2m_b_t
  );
end entity;

architecture a of cnn_accel_ofmap_dma is

  constant c_axi_data_width_bytes : positive := g_axi_data_width / 8;
  -- 'g_axi_data_width' is constrained to 'axi_pkg.axi_data_width_t' by the
  -- wrapped instance's own generic (a power-of-two AXI-legal width), so
  -- 'c_axi_data_width_bytes' is always a power of two too and this shift
  -- amount is exact -- see proposal doc section 3.3/section 6.
  constant c_bytes_to_beats_shift : natural := log2(c_axi_data_width_bytes);

  type state_t is (s_idle, s_zero_len, s_active, s_drain);
  signal state_q : state_t := s_idle;

  signal addr_q : unsigned(31 downto 0) := (others => '0');
  signal length_q : unsigned(31 downto 0) := (others => '0');
  signal expected_beats_q : unsigned(31 downto 0) := (others => '0');
  signal aw_issued_q : unsigned(31 downto 0) := (others => '0');
  signal bresp_acked_q : unsigned(31 downto 0) := (others => '0');
  signal error_latched_q : std_ulogic := '0';

  -- Deliberately not cleared by 'reset' -- see proposal doc section 4.
  signal outstanding_q : unsigned(7 downto 0) := (others => '0');
  signal outstanding_next_i : unsigned(7 downto 0);

  signal enable_i : std_ulogic;
  signal req_s2m_ready_i : std_ulogic;

  signal regs_down : dma_axi_write_simple_regs_down_t := dma_axi_write_simple_regs_down_init;
  signal regs_up : dma_axi_write_simple_regs_up_t;
  signal interrupt : std_ulogic;

  signal stream_ready_i : std_ulogic;
  signal stream_data_i : std_ulogic_vector(g_axi_data_width - 1 downto 0);

  signal axi_write_m2s : axi_write_m2s_t;
  signal axi_write_s2m : axi_write_s2m_t;

  signal aw_handshake_i : std_ulogic;
  signal b_handshake_i : std_ulogic;
  signal b_completes_request_i : std_ulogic;

begin

  ------------------------------------------------------------------------------
  -- Elaboration-time bus-width bound (decision S6 + decision D1, 2026-09-07).
  --
  -- The wrapped 'dma_axi_write_simple' core drives a full-width 'WSTRB' and
  -- has no partial-packet flush at any 'packet_length_beats', so it requires
  -- 'req.addr' and 'req.length' to be exact multiples of
  -- 'g_axi_data_width/8'. A violation does NOT report an error: the
  -- under-length final chunk is simply never issued, 'dma_done' never fires,
  -- and the layer hangs silently. That is the worst possible failure mode for
  -- a contract nothing else checks.
  --
  -- Decision S6 closes it statically rather than with a runtime check. Every
  -- activation DMA request is one whole channel-tiled plane
  -- ('[C/T][H][W][T]', T = 'cnn_accel_constant_activation_plane_channels'),
  -- so both 'addr' and 'length' are always multiples of T bytes. A bus of at
  -- most T bytes per beat is therefore aligned unconditionally, for every
  -- layer geometry, with no runtime comparison and no host-side contract to
  -- get wrong. Above that width alignment would depend on runtime descriptor
  -- fields ('out_width * out_height' even, base address aligned), which
  -- cannot be checked at elaboration at all -- hence a hard bound here
  -- instead of a weaker guarantee later.
  --
  -- 'severity failure' (not 'warning', unlike the recommendation-style assert
  -- in cnn_accel_conv_core): exceeding this bound is not a suboptimal
  -- configuration, it is a hang.
  ------------------------------------------------------------------------------

  assert g_axi_data_width <= cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_max_axi_data_width
    report "cnn_accel_ofmap_dma: g_axi_data_width (" & positive'image(g_axi_data_width) &
      ") exceeds the S6 activation-plane bound (" &
      integer'image(cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_max_axi_data_width) &
      " bits = " &
      integer'image(cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_activation_plane_channels) &
      " bytes/beat). Activation planes are T-byte granular, so a wider bus " &
      "makes addr/length alignment depend on runtime layer geometry, and an " &
      "unaligned request hangs (dma_done never fires) instead of erroring. " &
      "See doc/cnn_accel_arch.md 'Off-chip activation layout (decision S6)'."
    severity failure;

  ------------------------------------------------------------------------------
  -- Data-plane passthrough: stream in, AXI write channels out. No logic is
  -- added on either path beyond the field fan-out/fan-in itself -- see
  -- proposal doc section 5.
  ------------------------------------------------------------------------------

  stream_data_i <= s_stream_m2s.data(stream_data_i'range);
  s_stream_s2m.ready <= stream_ready_i;

  m_axi_aw_m2s <= axi_write_m2s.aw;
  axi_write_s2m.aw <= m_axi_aw_s2m;
  m_axi_w_m2s <= axi_write_m2s.w;
  axi_write_s2m.w <= m_axi_w_s2m;
  m_axi_b_m2s <= axi_write_m2s.b;
  axi_write_s2m.b <= m_axi_b_s2m;

  aw_handshake_i <= axi_write_m2s.aw.valid and axi_write_s2m.aw.ready;
  b_handshake_i <= axi_write_m2s.b.ready and axi_write_s2m.b.valid;
  b_completes_request_i <= b_handshake_i
    when state_q = s_active and bresp_acked_q + 1 = expected_beats_q else '0';


  ------------------------------------------------------------------------------
  -- Ring-buffer register-plane adaptation: degenerate single-shot buffer,
  -- one request at a time. See proposal doc section 3.2.
  ------------------------------------------------------------------------------

  regs_down.buffer_start_address <= std_ulogic_vector(
    resize(addr_q, regs_down.buffer_start_address'length)
  );
  -- One extra segment (= one AXI beat, since 'packet_length_beats => 1'
  -- below, see section 3.3) of padding beyond 'addr + length': this makes
  -- 'buffer_read_address = buffer_start_address' (below) a *correct*
  -- "whole region free" declaration instead of a fiction. See
  -- Implementation Notes in the proposal doc for why this replaced the
  -- original section-3.2 plan ('buffer_read_address <= addr + length').
  regs_down.buffer_end_address <= std_ulogic_vector(
    resize(addr_q + length_q + c_axi_data_width_bytes, regs_down.buffer_end_address'length)
  );
  -- Kept equal to 'buffer_start_address' at all times (never advanced) --
  -- required by 'ring_buffer_write_simple's own simulation-only assertion
  -- ("Initial read address should be start address"), checked on every
  -- 'enable' rising edge. Combined with the one-segment padding on
  -- 'buffer_end_address' above, the ring buffer's own "never more than
  -- size-1 segments outstanding" rule now permits exactly this request's
  -- 'expected_beats_q' segments before it self-blocks -- see
  -- Implementation Notes in the proposal doc.
  regs_down.buffer_read_address <= regs_down.buffer_start_address;
  regs_down.config.enable <= enable_i;
  -- Never masks/clears anything -- 'regs_up'/'interrupt' are not read at
  -- all (proposal doc section 3.5).
  regs_down.interrupt_mask <= (others => '0');
  regs_down.interrupt_status <= dma_axi_write_simple_interrupt_status_init;

  -- Belt-and-suspenders self-limit to exactly this request's segment
  -- count, agreeing with the ring buffer's own now-correct self-block
  -- (above): stops issuing new 'AW's once this request's beats have all
  -- been issued, without waiting for the ring buffer's internal state.
  enable_i <= '1' when state_q = s_active and aw_issued_q < expected_beats_q else '0';

  -- Combinational: always in step with 'state_q', so a reset that lands
  -- back in 's_idle' makes 'req_s2m.ready' high again the very next cycle.
  req_s2m_ready_i <= '1' when state_q = s_idle else '0';
  req_s2m.ready <= req_s2m_ready_i;

  -- Pulses: derived combinationally from the current state and this
  -- cycle's 'B' handshake, not registered -- see proposal doc section 3.5/
  -- section 5's state-machine description.
  dma_done <= '1' when state_q = s_zero_len else
              '1' when b_completes_request_i = '1' else
              '0';
  resp_error <= '1' when (
    b_completes_request_i = '1'
    and (error_latched_q = '1' or axi_write_s2m.b.resp /= axi_resp_okay)
  ) else '0';


  ------------------------------------------------------------------------------
  -- 'outstanding_q': physical AW-accepted-but-B-not-yet-seen transaction
  -- count. Updated unconditionally every cycle, including during 'reset' --
  -- it must never lose track of transactions actually in flight on the bus.
  -- See proposal doc section 4.
  ------------------------------------------------------------------------------

  outstanding_next_i <= outstanding_q + 1 when aw_handshake_i = '1' and b_handshake_i = '0' else
                        outstanding_q - 1 when aw_handshake_i = '0' and b_handshake_i = '1' else
                        outstanding_q;


  ------------------------------------------------------------------------------
  main : process(clk) is
  begin
    if rising_edge(clk) then
      outstanding_q <= outstanding_next_i;

      if reset = '1' then
        aw_issued_q <= (others => '0');
        bresp_acked_q <= (others => '0');
        error_latched_q <= '0';
        addr_q <= (others => '0');
        length_q <= (others => '0');
        expected_beats_q <= (others => '0');

        if outstanding_next_i = 0 then
          state_q <= s_idle;
        else
          -- At least one 'AW'-accepted transaction has not yet returned its
          -- 'BRESP'. Drain it before accepting a new request -- see
          -- proposal doc section 4, points 4-6.
          state_q <= s_drain;
        end if;
      else
        case state_q is
          when s_idle =>
            if req_m2s.valid = '1' then
              addr_q <= req_m2s.req.addr;
              length_q <= req_m2s.req.length;
              expected_beats_q <= shift_right(req_m2s.req.length, c_bytes_to_beats_shift);
              aw_issued_q <= (others => '0');
              bresp_acked_q <= (others => '0');
              error_latched_q <= '0';

              if shift_right(req_m2s.req.length, c_bytes_to_beats_shift) = 0 then
                -- 'length = 0' is legal and trivially "fully written" with
                -- zero bursts (proposal doc section 5's state machine).
                state_q <= s_zero_len;
              else
                state_q <= s_active;
              end if;
            end if;

          when s_zero_len =>
            state_q <= s_idle;

          when s_active =>
            if aw_handshake_i = '1' then
              aw_issued_q <= aw_issued_q + 1;
            end if;

            if b_handshake_i = '1' then
              bresp_acked_q <= bresp_acked_q + 1;

              if axi_write_s2m.b.resp /= axi_resp_okay then
                error_latched_q <= '1';
              end if;

              if bresp_acked_q + 1 = expected_beats_q then
                state_q <= s_idle;
              end if;
            end if;

          when s_drain =>
            if outstanding_next_i = 0 then
              state_q <= s_idle;
            end if;
        end case;
      end if;
    end if;
  end process;


  ------------------------------------------------------------------------------
  dma_axi_write_simple_inst : entity dma_axi_write_simple.dma_axi_write_simple
    generic map (
      address_width => g_axi_addr_width,
      stream_data_width => g_axi_data_width,
      axi_data_width => g_axi_data_width,
      packet_length_beats => 1,
      enable_axi3 => false,
      write_done_aggregate_count => 1,
      write_done_aggregate_ticks => 1
    )
    port map (
      clk => clk,
      --
      stream_ready => stream_ready_i,
      stream_valid => s_stream_m2s.valid,
      stream_data => stream_data_i,
      --
      regs_up => regs_up,
      regs_down => regs_down,
      interrupt => interrupt,
      --
      axi_write_m2s => axi_write_m2s,
      axi_write_s2m => axi_write_s2m
    );

end architecture;
