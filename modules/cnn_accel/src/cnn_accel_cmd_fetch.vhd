library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_v2_pkg.all;
use cnn_accel.cnn_accel_isa_pkg.c_instr_word_bytes;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- ISA v2.0 descriptor fetch unit (doc/cnn_accel_top_v2_arch.md section 2.1,
-- "cnn_accel_cmd_fetch"). Issues one 'dma_req_m2s_t' read job of exactly
-- 'c_instr_word_bytes' (the generated constant, never a literal 64 here) at
-- a given descriptor byte address, assembles the returned little-endian
-- AXI4-Stream beats into one 'c_instr_word_bytes'-byte (512-bit) word,
-- decodes it with 'decode_desc_v2' (cnn_accel_v2_pkg), and hands the
-- decoded 'desc_v2_t' plus the descriptor's own fetch address ('pc') to
-- the consumer (cnn_accel_cmd_proc) through a valid/ready handshake.
--
-- This entity owns no AXI4 master port of its own (unlike
-- 'cnn_accel_axi_read_dma', which it does not instantiate): per the
-- top-level diagram (arch doc section 2), the actual
-- 'cnn_accel_axi_read_dma' instance for instruction fetch sits one level
-- up (in 'cnn_accel_top'/'cnn_accel_cmd_proc'), wired to this entity's
-- 'instr_req_m2s'/'instr_req_s2m' request port and 's_instr_stream_m2s'/
-- 's_instr_stream_s2m' AXI4-Stream port exactly as a
-- 'cnn_accel_axi_read_dma' instance's 'req_m2s'/'req_s2m'/'m_stream_m2s'/
-- 'm_stream_s2m' would be driven/consumed by any other caller. This
-- entity additionally consumes that same DMA instance's 'dma_done'/
-- 'resp_error' pulses directly (not carried on the stream -- see
-- 'cnn_accel_axi_read_dma's own header comment) to know when the whole
-- burst has drained and whether any 'RRESP' in it was non-OKAY.
--
-- FSM: 'IDLE -> REQ -> COLLECT -> PRESENT -> IDLE' on success,
-- '-> ERROR -> IDLE' on an AXI error or a watchdog timeout. Modeled after
-- the v1 'cnn_accel_sequencer' requirement's FETCH/DECODE split
-- (doc/cnn_accel_sequencer_req.md), adapted to the v2.0 'desc_v2_t'/
-- 'decode_desc_v2' decode step.
--
-- Restart / chaining: 'start' is only sampled in 'IDLE', so once
-- 'PRESENT' has been acknowledged ('desc_ready') this entity is
-- immediately ready for the next 'start'/'addr' pair -- the caller drives
-- 'addr <= desc.next_instr_addr' for a chained fetch, exactly like the v1
-- sequencer's 'pc <= next_instr_addr' step.
--
-- Back-pressure: the incoming AXI4-Stream is only ever accepted (this
-- entity's 's_instr_stream_s2m.ready') while collecting beats in
-- 'COLLECT', so a beat is captured if and only if both 'valid' and
-- 'ready' are seen together that cycle -- no beat is ever dropped or
-- counted twice. The decoded descriptor output likewise holds 'PRESENT'
-- (desc/desc_valid stable) for as many cycles as the consumer needs
-- before raising 'desc_ready'.
--
-- Error handling: an AXI error surfaced by the DMA instance
-- ('instr_resp_error' pulsing alongside 'instr_dma_done') reports
-- 'c_err_axi'; if 'g_timeout_cycles > 0' and no 'instr_dma_done' has
-- arrived within that many cycles of 'start' being accepted (covering
-- both the request-acceptance wait in 'REQ' and the beat-collection wait
-- in 'COLLECT'), this entity abandons the fetch and reports
-- 'c_err_timeout' instead of hanging forever -- 'g_timeout_cycles = 0'
-- (the default) disables the watchdog entirely. Either error path
-- deterministically returns to 'IDLE' one cycle later, ready for the next
-- 'start'.
--
-- Reset: synchronous active-high 'reset' ('reset_internal' at the IP top
-- level) is relied on to abort an in-flight fetch cleanly (no separate
-- 'soft_reset' port) -- the same convention 'cnn_accel_axi_read_dma'
-- itself uses. A fetch aborted by 'reset' simply never reaches 'PRESENT';
-- any beats the upstream DMA instance was already mid-burst on are its
-- own concern (its documented stale-beat draining), not this entity's,
-- since a fresh 'start' after 'reset' always begins a brand new request.
entity cnn_accel_cmd_fetch is
  generic (
    -- AXI4-Stream beat width for the instruction-fetch DMA instance this
    -- entity's request/stream ports are wired to. Must match that
    -- instance's own 'g_axi_data_width'.
    g_axi_data_width : positive := 64;
    -- Watchdog bound, in clock cycles, from 'start' being accepted until
    -- 'instr_dma_done' must arrive. '0' (default) disables the watchdog.
    g_timeout_cycles : natural := 0
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';

    --# {{}}
    -- Begin a fetch of the descriptor at byte address 'addr'. Sampled
    -- only while idle; re-startable immediately after a fetch's 'PRESENT'
    -- has been acknowledged, for chained descriptors.
    start : in std_ulogic;
    addr : in unsigned(31 downto 0);

    --# {{}}
    -- Request port to this entity's instruction-fetch 'cnn_accel_axi_read_dma'
    -- instance (owned one level up -- see entity-level comment).
    instr_req_m2s : out dma_req_m2s_t :=
      (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
    instr_req_s2m : in dma_req_s2m_t;
    -- AXI4-Stream instruction bytes from that same DMA instance.
    s_instr_stream_m2s : in axi_stream_m2s_t;
    s_instr_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;
    -- That DMA instance's own 'dma_done'/'resp_error' pulses (not carried
    -- on the stream -- see entity-level comment).
    instr_dma_done : in std_ulogic;
    instr_resp_error : in std_ulogic;

    --# {{}}
    -- Decoded descriptor output handshake.
    desc : out desc_v2_t := desc_v2_init;
    -- The descriptor's own fetch address ("PC"), stable while 'desc_valid'.
    pc : out unsigned(31 downto 0) := (others => '0');
    desc_valid : out std_ulogic := '0';
    desc_ready : in std_ulogic;

    --# {{}}
    -- One-cycle error pulse; 'error_code' is 'c_err_axi' or 'c_err_timeout'
    -- and is only meaningful on the same cycle as 'error'.
    error : out std_ulogic := '0';
    error_code : out err_code_t := c_err_none
  );
end entity cnn_accel_cmd_fetch;

architecture a of cnn_accel_cmd_fetch is

  constant c_bytes_per_beat : positive := g_axi_data_width / 8;
  constant c_bits_per_beat : positive := 8 * c_bytes_per_beat;
  constant c_beats_per_word : positive := c_instr_word_bytes / c_bytes_per_beat;
  constant c_word_bits : positive := 8 * c_instr_word_bytes;

  type state_t is (s_idle, s_req, s_collect, s_present, s_error);
  signal state_q : state_t := s_idle;

  signal pc_q : unsigned(31 downto 0) := (others => '0');
  signal beat_count_q : natural range 0 to c_beats_per_word := 0;
  signal word_q : std_ulogic_vector(c_word_bits - 1 downto 0) := (others => '0');
  signal error_code_q : err_code_t := c_err_none;
  -- Watchdog: counts cycles since 'start' was accepted; only meaningful
  -- (and only ever incremented) while 'g_timeout_cycles > 0'.
  signal timeout_count_q : natural := 0;

begin

  ------------------------------------------------------------------------
  -- Combinational outputs, all pure functions of registered state: no
  -- separate "eager" register is needed for 'desc'/'pc' because 'word_q'
  -- already holds the fully-assembled word by the time 'state_q' reads
  -- as 's_present' (both update together on the same clock edge that
  -- captures the final beat and advances the FSM -- see the 'COLLECT'
  -- branch below).
  ------------------------------------------------------------------------

  desc <= decode_desc_v2(word_q) when state_q = s_present else desc_v2_init;
  desc_valid <= '1' when state_q = s_present else '0';
  pc <= pc_q;

  error <= '1' when state_q = s_error else '0';
  error_code <= error_code_q when state_q = s_error else c_err_none;

  instr_req_m2s.valid <= '1' when state_q = s_req else '0';
  instr_req_m2s.req.addr <= pc_q;
  instr_req_m2s.req.length <= to_unsigned(c_instr_word_bytes, 32);

  s_instr_stream_s2m.ready <= '1' when state_q = s_collect else '0';

  ------------------------------------------------------------------------
  -- FSM: state, beat assembly, watchdog.
  ------------------------------------------------------------------------

  fsm : process(clk)
    variable word_next : std_ulogic_vector(c_word_bits - 1 downto 0);
    variable timed_out : boolean;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        state_q <= s_idle;
        beat_count_q <= 0;
        word_q <= (others => '0');
        timeout_count_q <= 0;
        error_code_q <= c_err_none;
      else
        timed_out := g_timeout_cycles > 0 and timeout_count_q >= g_timeout_cycles;

        case state_q is
          when s_idle =>
            if start = '1' then
              pc_q <= addr;
              beat_count_q <= 0;
              word_q <= (others => '0');
              timeout_count_q <= 0;
              state_q <= s_req;
            end if;

          when s_req =>
            if timed_out then
              error_code_q <= c_err_timeout;
              state_q <= s_error;
            elsif instr_req_s2m.ready = '1' then
              timeout_count_q <= timeout_count_q + 1;
              state_q <= s_collect;
            else
              timeout_count_q <= timeout_count_q + 1;
            end if;

          when s_collect =>
            if timed_out then
              error_code_q <= c_err_timeout;
              state_q <= s_error;
            else
              timeout_count_q <= timeout_count_q + 1;

              if s_instr_stream_m2s.valid = '1' then
                -- Constant-bound loop with the beat index as a per-lane
                -- enable (house style: cnn_accel_weight_buffer's fill
                -- process), not a dynamically-based slice -- see that
                -- entity's comment for why. Little-endian: beat 0 is the
                -- descriptor's lowest byte offset, landing in 'word_q's
                -- low bits.
                word_next := word_q;
                for beat in 0 to c_beats_per_word - 1 loop
                  if beat_count_q = beat then
                    word_next(c_bits_per_beat * (beat + 1) - 1 downto c_bits_per_beat * beat) :=
                      s_instr_stream_m2s.data(c_bits_per_beat - 1 downto 0);
                  end if;
                end loop;
                word_q <= word_next;

                if beat_count_q < c_beats_per_word - 1 then
                  beat_count_q <= beat_count_q + 1;
                end if;
              end if;

              if instr_dma_done = '1' then
                if instr_resp_error = '1' then
                  error_code_q <= c_err_axi;
                  state_q <= s_error;
                else
                  state_q <= s_present;
                end if;
              end if;
            end if;

          when s_present =>
            if desc_ready = '1' then
              state_q <= s_idle;
            end if;

          when s_error =>
            -- One-cycle pulse (see the concurrent 'error'/'error_code'
            -- assignments above); unconditionally back to idle.
            state_q <= s_idle;

        end case;
      end if;
    end if;
  end process;

end architecture a;
