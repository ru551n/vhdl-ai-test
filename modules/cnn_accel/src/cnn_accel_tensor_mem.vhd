library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.ceil_log2;
use math.math_pkg.is_power_of_two;

-- Local tensor scratchpad (doc/cnn_accel_top_v2_arch.md section 4):
-- 'g_num_banks' independent banks of 'g_bank_words' x 'g_data_width'-bit
-- words, exposed as 2 AXI-Stream write channels ('w0'/'w1') and 2
-- AXI-Stream read channels ('r0'/'r1'), each fronted by the same
-- '(addr, length)'-in-bytes 'dma_req_m2s_t'/'dma_req_s2m_t' request idiom
-- as 'cnn_accel_axi_read_dma'/'cnn_accel_ofmap_dma', so engines are
-- agnostic to whether their data came from DDR or the scratchpad.
--
-- Addressing: 'addr'/'length' are byte values, guaranteed 8-byte aligned
-- by the caller (asserted, not corrected). 'word_addr = addr / 8'; since
-- 'g_bank_words' is required to be a power of two (asserted below),
-- 'bank = word_addr / g_bank_words' and 'offset = word_addr mod
-- g_bank_words' both synthesize as plain bit slices of 'word_addr', not
-- as dividers, even though they are spelled with '/'/'mod' below for
-- readability -- see 'decode_addr'.
--
-- A request that would cross a bank boundary (offset + beats >
-- g_bank_words) is a caller bug: this entity never corrupts a
-- neighbouring bank's contents in that case, but it also does not try to
-- "do the right thing" -- it reports the error (simulation-only 'assert',
-- severity 'error' so the run keeps going and the bug is visible in the
-- log) and silently clamps the transfer length to what still fits in the
-- targeted bank. Same treatment for a totally out-of-range bank index
-- (decodes to 'bank >= g_num_banks'): asserted, then wrapped modulo
-- 'g_num_banks' so no array bound is ever violated.
--
-- Arbitration: each bank has exactly one physical write port and one
-- physical read port (simple dual-port RAM, 1-cycle synchronous read
-- latency, no read-during-write forwarding). Both write channels (and,
-- separately, both read channels) may target the same bank on the same
-- cycle; a per-bank, per-direction round-robin arbiter (a single
-- "who-won-last-time" bit) grants exactly one of them and back-pressures
-- (deasserts 'ready' towards) the other, alternating on sustained
-- contention so neither channel can starve the other. Channels aimed at
-- *different* banks never contend and both proceed at full rate in the
-- same cycle, since each bank's port is independent.
--
-- Backpressure: a write channel's 's_w*_s2m.ready' is asserted only on
-- the cycle it is actually granted the bank write port, so a beat is
-- consumed if and only if it is written -- no drops, no duplicates. A
-- read channel only issues a bank read when it has no unconsumed beat
-- already waiting on its own output register (or that beat is being
-- accepted this very cycle), and the beat that comes back one cycle
-- later is captured into that channel's own private output register
-- (never read directly out of the shared per-bank read-data register),
-- so a slow consumer on one channel can never cause another channel's
-- in-flight read to be silently overwritten.
entity cnn_accel_tensor_mem is
  generic (
    g_num_banks : positive := 2;
    g_bank_words : positive := 1024;
    g_data_width : positive := 64
  );
  port (
    clk : in std_ulogic;
    reset : in std_ulogic := '0';

    --# {{}}
    -- Write channel 0.
    w0_req_m2s : in dma_req_m2s_t;
    w0_req_s2m : out dma_req_s2m_t;
    s_w0_m2s : in axi_stream_m2s_t;
    s_w0_s2m : out axi_stream_s2m_t;
    w0_done : out std_ulogic := '0';

    --# {{}}
    -- Write channel 1.
    w1_req_m2s : in dma_req_m2s_t;
    w1_req_s2m : out dma_req_s2m_t;
    s_w1_m2s : in axi_stream_m2s_t;
    s_w1_s2m : out axi_stream_s2m_t;
    w1_done : out std_ulogic := '0';

    --# {{}}
    -- Read channel 0.
    r0_req_m2s : in dma_req_m2s_t;
    r0_req_s2m : out dma_req_s2m_t;
    m_r0_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_r0_s2m : in axi_stream_s2m_t;
    r0_done : out std_ulogic := '0';

    --# {{}}
    -- Read channel 1.
    r1_req_m2s : in dma_req_m2s_t;
    r1_req_s2m : out dma_req_s2m_t;
    m_r1_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_r1_s2m : in axi_stream_s2m_t;
    r1_done : out std_ulogic := '0'
  );
end entity cnn_accel_tensor_mem;

architecture a of cnn_accel_tensor_mem is

  constant c_bytes_per_word : positive := g_data_width / 8;

  subtype bank_idx_t is natural range 0 to g_num_banks - 1;
  subtype word_off_t is natural range 0 to g_bank_words - 1;
  subtype beat_cnt_t is natural range 0 to g_bank_words;
  subtype data_word_t is std_ulogic_vector(g_data_width - 1 downto 0);
  subtype owner_t is natural range 0 to 1;

  type addr_decode_t is record
    bank   : bank_idx_t;
    offset : word_off_t;
  end record;

  -- 'word_addr / g_bank_words' and 'word_addr mod g_bank_words' both
  -- reduce to a bit slice of 'word_addr' because 'g_bank_words' is a
  -- power of two (asserted below); spelled with '/'/'mod' only for
  -- readability. Out-of-range bank indices (caller bug) are reported and
  -- wrapped, never corrupting an unintended bank via an out-of-bounds
  -- array access.
  function to_sl(value : boolean) return std_ulogic is
  begin
    if value then
      return '1';
    else
      return '0';
    end if;
  end function;

  function decode_addr(addr : unsigned(31 downto 0)) return addr_decode_t is
    variable word_addr : natural;
    variable bank_raw : natural;
    variable result : addr_decode_t;
  begin
    word_addr := to_integer(addr) / c_bytes_per_word;
    bank_raw := word_addr / g_bank_words;

    assert bank_raw < g_num_banks
      report "cnn_accel_tensor_mem: address decodes to bank " & natural'image(bank_raw) &
        ", outside g_num_banks=" & natural'image(g_num_banks) &
        " (validation should have rejected this address before dispatch)"
      severity error;

    result.bank := bank_raw mod g_num_banks;
    result.offset := word_addr mod g_bank_words;
    return result;
  end function;

  type bank_mem_t is array (0 to g_bank_words - 1) of data_word_t;
  type bank_mem_arr_t is array (natural range <>) of bank_mem_t;
  signal bank_ram : bank_mem_arr_t(0 to g_num_banks - 1);

  -- Per-bank write arbitration: which channel (0 or 1) won the write
  -- port on the previous contended cycle, so the next contended cycle
  -- alternates to the other one.
  signal w_last_granted : owner_t := 0;
  type owner_arr_t is array (natural range <>) of owner_t;
  signal w_last_granted_bank : owner_arr_t(0 to g_num_banks - 1) := (others => 0);
  signal r_last_granted_bank : owner_arr_t(0 to g_num_banks - 1) := (others => 0);

  -- Per-bank registered read result (1-cycle synchronous-read latency),
  -- tagged with which channel issued the read that produced it, so that
  -- channel (and only that channel) can capture it into its own private
  -- output register on the following cycle.
  type data_word_arr_t is array (natural range <>) of data_word_t;
  signal bank_rd_data_q : data_word_arr_t(0 to g_num_banks - 1) := (others => (others => '0'));
  signal bank_rd_valid_q : std_ulogic_vector(0 to g_num_banks - 1) := (others => '0');
  signal bank_rd_last_q : std_ulogic_vector(0 to g_num_banks - 1) := (others => '0');
  signal bank_rd_owner_q : owner_arr_t(0 to g_num_banks - 1) := (others => 0);

  -- Write channel 0/1 state.
  signal w0_busy, w1_busy : std_ulogic := '0';
  signal w0_bank, w1_bank : bank_idx_t := 0;
  signal w0_offset, w1_offset : word_off_t := 0;
  signal w0_beats_left, w1_beats_left : beat_cnt_t := 0;
  signal w0_done_q, w1_done_q : std_ulogic := '0';

  -- Combinational write arbitration.
  signal w0_wants, w1_wants : std_ulogic;
  signal w0_grant, w1_grant : std_ulogic;

  -- Read channel 0/1 state.
  signal r0_busy, r1_busy : std_ulogic := '0';
  signal r0_bank, r1_bank : bank_idx_t := 0;
  signal r0_offset_next, r1_offset_next : word_off_t := 0;
  signal r0_beats_to_issue, r1_beats_to_issue : beat_cnt_t := 0;
  signal r0_out_valid, r1_out_valid : std_ulogic := '0';
  signal r0_out_last, r1_out_last : std_ulogic := '0';
  signal r0_out_data, r1_out_data : data_word_t := (others => '0');

  -- Combinational read arbitration / issue gating.
  signal r0_can_issue, r1_can_issue : std_ulogic;
  signal r0_grant, r1_grant : std_ulogic;
  signal r0_capture, r1_capture : std_ulogic;

begin

  ------------------------------------------------------------------------
  -- Generic sanity checks.
  ------------------------------------------------------------------------

  assert is_power_of_two(g_bank_words)
    report "cnn_accel_tensor_mem: g_bank_words must be a power of two, got " &
      positive'image(g_bank_words)
    severity failure;

  ------------------------------------------------------------------------
  -- Request-accept handshakes: a channel can accept a new request only
  -- while idle.
  ------------------------------------------------------------------------

  w0_req_s2m.ready <= not w0_busy;
  w1_req_s2m.ready <= not w1_busy;
  r0_req_s2m.ready <= not r0_busy;
  r1_req_s2m.ready <= not r1_busy;

  ------------------------------------------------------------------------
  -- Write-side round-robin arbitration (per pair of channels contending
  -- for the same bank; channels on different banks never contend).
  ------------------------------------------------------------------------

  w0_wants <= w0_busy and s_w0_m2s.valid;
  w1_wants <= w1_busy and s_w1_m2s.valid;

  write_arbitrate : process(all)
  begin
    if w0_wants = '1' and w1_wants = '1' and w0_bank = w1_bank then
      -- Contention on the same bank: give it to whichever channel did not
      -- win last time on that bank.
      if w_last_granted_bank(w0_bank) = 0 then
        w0_grant <= '0';
        w1_grant <= '1';
      else
        w0_grant <= '1';
        w1_grant <= '0';
      end if;
    else
      w0_grant <= w0_wants;
      w1_grant <= w1_wants;
    end if;
  end process;

  s_w0_s2m.ready <= w0_grant;
  s_w1_s2m.ready <= w1_grant;

  ------------------------------------------------------------------------
  -- Per-bank physical write port: performs the actual RAM write for
  -- whichever channel was granted this bank this cycle, and updates that
  -- bank's round-robin memory.
  ------------------------------------------------------------------------

  bank_write_gen : for b in 0 to g_num_banks - 1 generate
    bank_write : process(clk)
    begin
      if rising_edge(clk) then
        if w0_grant = '1' and w0_bank = b then
          bank_ram(b)(w0_offset) <= s_w0_m2s.data(g_data_width - 1 downto 0);
          w_last_granted_bank(b) <= 0;
        elsif w1_grant = '1' and w1_bank = b then
          bank_ram(b)(w1_offset) <= s_w1_m2s.data(g_data_width - 1 downto 0);
          w_last_granted_bank(b) <= 1;
        end if;
      end if;
    end process;
  end generate;

  ------------------------------------------------------------------------
  -- Write channel 0/1 FSMs: accept a request while idle, otherwise
  -- consume one granted beat per cycle and pulse 'done' one cycle after
  -- the last word is written.
  ------------------------------------------------------------------------

  write_fsm_0 : process(clk)
    variable decode : addr_decode_t;
    variable raw_beats, beats : natural;
  begin
    if rising_edge(clk) then
      w0_done_q <= '0';

      if reset = '1' then
        w0_busy <= '0';
      elsif w0_busy = '1' then
        if w0_grant = '1' then
          if w0_beats_left = 1 then
            w0_busy <= '0';
            w0_done_q <= '1';
          else
            w0_offset <= w0_offset + 1;
            w0_beats_left <= w0_beats_left - 1;
          end if;
        end if;
      elsif w0_req_m2s.valid = '1' then
        assert w0_req_m2s.req.addr(2 downto 0) = "000" and
          w0_req_m2s.req.length(2 downto 0) = "000"
          report "cnn_accel_tensor_mem: w0 request not 8-byte aligned" severity failure;

        decode := decode_addr(w0_req_m2s.req.addr);
        raw_beats := to_integer(w0_req_m2s.req.length) / c_bytes_per_word;

        if decode.offset + raw_beats > g_bank_words then
          assert false
            report "cnn_accel_tensor_mem: w0 request crosses a bank boundary; clamping"
            severity error;
          beats := g_bank_words - decode.offset;
        else
          beats := raw_beats;
        end if;

        if beats = 0 then
          w0_done_q <= '1';
        else
          w0_busy <= '1';
          w0_bank <= decode.bank;
          w0_offset <= decode.offset;
          w0_beats_left <= beats;
        end if;
      end if;
    end if;
  end process;

  write_fsm_1 : process(clk)
    variable decode : addr_decode_t;
    variable raw_beats, beats : natural;
  begin
    if rising_edge(clk) then
      w1_done_q <= '0';

      if reset = '1' then
        w1_busy <= '0';
      elsif w1_busy = '1' then
        if w1_grant = '1' then
          if w1_beats_left = 1 then
            w1_busy <= '0';
            w1_done_q <= '1';
          else
            w1_offset <= w1_offset + 1;
            w1_beats_left <= w1_beats_left - 1;
          end if;
        end if;
      elsif w1_req_m2s.valid = '1' then
        assert w1_req_m2s.req.addr(2 downto 0) = "000" and
          w1_req_m2s.req.length(2 downto 0) = "000"
          report "cnn_accel_tensor_mem: w1 request not 8-byte aligned" severity failure;

        decode := decode_addr(w1_req_m2s.req.addr);
        raw_beats := to_integer(w1_req_m2s.req.length) / c_bytes_per_word;

        if decode.offset + raw_beats > g_bank_words then
          assert false
            report "cnn_accel_tensor_mem: w1 request crosses a bank boundary; clamping"
            severity error;
          beats := g_bank_words - decode.offset;
        else
          beats := raw_beats;
        end if;

        if beats = 0 then
          w1_done_q <= '1';
        else
          w1_busy <= '1';
          w1_bank <= decode.bank;
          w1_offset <= decode.offset;
          w1_beats_left <= beats;
        end if;
      end if;
    end if;
  end process;

  w0_done <= w0_done_q;
  w1_done <= w1_done_q;

  ------------------------------------------------------------------------
  -- Read-side round-robin arbitration. A channel may issue a bank read
  -- only if it has beats left to issue and its own output register is
  -- free (or being drained this very cycle), so it never has more than
  -- one beat in flight.
  ------------------------------------------------------------------------

  r0_can_issue <= r0_busy and to_sl(r0_beats_to_issue > 0) and
    (not r0_out_valid or m_r0_s2m.ready);
  r1_can_issue <= r1_busy and to_sl(r1_beats_to_issue > 0) and
    (not r1_out_valid or m_r1_s2m.ready);

  read_arbitrate : process(all)
  begin
    if r0_can_issue = '1' and r1_can_issue = '1' and r0_bank = r1_bank then
      if r_last_granted_bank(r0_bank) = 0 then
        r0_grant <= '0';
        r1_grant <= '1';
      else
        r0_grant <= '1';
        r1_grant <= '0';
      end if;
    else
      r0_grant <= r0_can_issue;
      r1_grant <= r1_can_issue;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Per-bank physical read port: issues the read for whichever channel
  -- was granted this bank this cycle, and tags the 1-cycle-later result
  -- with its owner so only that channel captures it.
  ------------------------------------------------------------------------

  bank_read_gen : for b in 0 to g_num_banks - 1 generate
    bank_read : process(clk)
    begin
      if rising_edge(clk) then
        if r0_grant = '1' and r0_bank = b then
          bank_rd_data_q(b) <= bank_ram(b)(r0_offset_next);
          bank_rd_owner_q(b) <= 0;
          bank_rd_valid_q(b) <= '1';
          if r0_beats_to_issue = 1 then
            bank_rd_last_q(b) <= '1';
          else
            bank_rd_last_q(b) <= '0';
          end if;
          r_last_granted_bank(b) <= 0;
        elsif r1_grant = '1' and r1_bank = b then
          bank_rd_data_q(b) <= bank_ram(b)(r1_offset_next);
          bank_rd_owner_q(b) <= 1;
          bank_rd_valid_q(b) <= '1';
          if r1_beats_to_issue = 1 then
            bank_rd_last_q(b) <= '1';
          else
            bank_rd_last_q(b) <= '0';
          end if;
          r_last_granted_bank(b) <= 1;
        else
          bank_rd_valid_q(b) <= '0';
        end if;
      end if;
    end process;
  end generate;

  r0_capture <= bank_rd_valid_q(r0_bank) and to_sl(bank_rd_owner_q(r0_bank) = 0);
  r1_capture <= bank_rd_valid_q(r1_bank) and to_sl(bank_rd_owner_q(r1_bank) = 1);

  ------------------------------------------------------------------------
  -- Read channel 0/1 FSMs.
  ------------------------------------------------------------------------

  read_fsm_0 : process(clk)
    variable decode : addr_decode_t;
    variable raw_beats, beats : natural;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        r0_busy <= '0';
        r0_out_valid <= '0';
      else
        if r0_capture = '1' then
          r0_out_valid <= '1';
          r0_out_data <= bank_rd_data_q(r0_bank);
          r0_out_last <= bank_rd_last_q(r0_bank);
        elsif r0_out_valid = '1' and m_r0_s2m.ready = '1' then
          r0_out_valid <= '0';
        end if;

        if r0_out_valid = '1' and r0_out_last = '1' and m_r0_s2m.ready = '1' then
          r0_busy <= '0';
        end if;

        if r0_grant = '1' then
          if r0_beats_to_issue = 1 then
            r0_beats_to_issue <= 0;
          else
            r0_offset_next <= r0_offset_next + 1;
            r0_beats_to_issue <= r0_beats_to_issue - 1;
          end if;
        end if;

        if r0_busy = '0' and r0_req_m2s.valid = '1' then
          assert r0_req_m2s.req.addr(2 downto 0) = "000" and
            r0_req_m2s.req.length(2 downto 0) = "000"
            report "cnn_accel_tensor_mem: r0 request not 8-byte aligned" severity failure;

          decode := decode_addr(r0_req_m2s.req.addr);
          raw_beats := to_integer(r0_req_m2s.req.length) / c_bytes_per_word;

          if decode.offset + raw_beats > g_bank_words then
            assert false
              report "cnn_accel_tensor_mem: r0 request crosses a bank boundary; clamping"
              severity error;
            beats := g_bank_words - decode.offset;
          else
            beats := raw_beats;
          end if;

          if beats /= 0 then
            r0_busy <= '1';
            r0_bank <= decode.bank;
            r0_offset_next <= decode.offset;
            r0_beats_to_issue <= beats;
          end if;
        end if;
      end if;
    end if;
  end process;

  read_fsm_1 : process(clk)
    variable decode : addr_decode_t;
    variable raw_beats, beats : natural;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        r1_busy <= '0';
        r1_out_valid <= '0';
      else
        if r1_capture = '1' then
          r1_out_valid <= '1';
          r1_out_data <= bank_rd_data_q(r1_bank);
          r1_out_last <= bank_rd_last_q(r1_bank);
        elsif r1_out_valid = '1' and m_r1_s2m.ready = '1' then
          r1_out_valid <= '0';
        end if;

        if r1_out_valid = '1' and r1_out_last = '1' and m_r1_s2m.ready = '1' then
          r1_busy <= '0';
        end if;

        if r1_grant = '1' then
          if r1_beats_to_issue = 1 then
            r1_beats_to_issue <= 0;
          else
            r1_offset_next <= r1_offset_next + 1;
            r1_beats_to_issue <= r1_beats_to_issue - 1;
          end if;
        end if;

        if r1_busy = '0' and r1_req_m2s.valid = '1' then
          assert r1_req_m2s.req.addr(2 downto 0) = "000" and
            r1_req_m2s.req.length(2 downto 0) = "000"
            report "cnn_accel_tensor_mem: r1 request not 8-byte aligned" severity failure;

          decode := decode_addr(r1_req_m2s.req.addr);
          raw_beats := to_integer(r1_req_m2s.req.length) / c_bytes_per_word;

          if decode.offset + raw_beats > g_bank_words then
            assert false
              report "cnn_accel_tensor_mem: r1 request crosses a bank boundary; clamping"
              severity error;
            beats := g_bank_words - decode.offset;
          else
            beats := raw_beats;
          end if;

          if beats /= 0 then
            r1_busy <= '1';
            r1_bank <= decode.bank;
            r1_offset_next <= decode.offset;
            r1_beats_to_issue <= beats;
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Read channel outputs. 'axi_stream_m2s_t.data' is a fixed 128-bit
  -- field (this project's 'axi_stream_pkg' has no 'strb'/'keep' members,
  -- matching 'cnn_accel_axi_read_dma'/'cnn_accel_ofmap_dma' usage
  -- elsewhere in this codebase, and every beat here is a full
  -- 'g_data_width'-bit word, so there is no partial-beat case to flag);
  -- only the low 'g_data_width' bits carry the payload, zero-extended.
  ------------------------------------------------------------------------

  m_r0_m2s.valid <= r0_out_valid;
  m_r0_m2s.last <= r0_out_last;
  m_r0_m2s.user <= (others => '0');
  m_r0_m2s.data <= std_ulogic_vector(resize(unsigned(r0_out_data), axi_stream_data_sz));

  m_r1_m2s.valid <= r1_out_valid;
  m_r1_m2s.last <= r1_out_last;
  m_r1_m2s.user <= (others => '0');
  m_r1_m2s.data <= std_ulogic_vector(resize(unsigned(r1_out_data), axi_stream_data_sz));

  r0_done <= r0_out_valid and r0_out_last and m_r0_s2m.ready;
  r1_done <= r1_out_valid and r1_out_last and m_r1_s2m.ready;

end architecture a;
