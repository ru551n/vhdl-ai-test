library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.memory_pkg.all;
use vunit_lib.axi_slave_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library bfm;
use bfm.stall_bfm_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- VUnit-5 testbench for cnn_accel_axi_read_dma. See
-- modules/cnn_accel/doc/cnn_accel_axi_read_dma_req.md and
-- modules/cnn_accel/doc/cnn_accel_axi_read_dma_proposal.md section 10 for
-- the verification plan.
--
-- The AXI4 read slave side is bfm.axi_read_slave (hdl-modules wrapper
-- around VUnit's axi_read_slave verification component + memory model),
-- with randomized AR/R stalling and response latency. VUnit's slave always
-- answers RRESP=OKAY (vunit/vhdl/verification_components/src/
-- axi_read_slave.vhd), so the 'resp_error' test uses a passive wire-level
-- override of the RRESP field between the BFM and the DUT for one chosen
-- beat -- the BFM still owns every handshake and data byte.
-- The AXI4-Stream consumer side reuses bfm.axi_stream_slave (packet-level
-- data/last checking plus randomized backpressure), since one whole DMA
-- request maps to exactly one AXI4-Stream packet.
entity tb_cnn_accel_axi_read_dma is
  generic (runner_cfg : string);
end entity tb_cnn_accel_axi_read_dma;

architecture tb of tb_cnn_accel_axi_read_dma is

  constant c_axi_addr_width : positive := 32;
  constant c_axi_data_width : positive := 32;
  constant c_axi_id_width : natural := 4;
  constant c_bytes_per_beat : positive := c_axi_data_width / 8;

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '0';

  signal req_m2s : dma_req_m2s_t := (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal req_s2m : dma_req_s2m_t;

  signal dma_done : std_ulogic;
  signal resp_error : std_ulogic;

  signal m_axi_ar_m2s : axi_m2s_a_t := axi_m2s_a_init;
  signal m_axi_ar_s2m : axi_s2m_a_t := axi_s2m_a_init;
  signal m_axi_r_m2s : axi_m2s_r_t := axi_m2s_r_init;
  signal m_axi_r_s2m : axi_s2m_r_t := axi_s2m_r_init;

  -- Bundled view of the DUT's AR+R ports for the BFM, and the BFM's own
  -- (always-OKAY) S2M response before the RRESP override below.
  signal axi_read_m2s : axi_read_m2s_t := axi_read_m2s_init;
  signal axi_read_s2m_bfm : axi_read_s2m_t := axi_read_s2m_init;

  -- One flat region covering every address any test below touches (highest
  -- is 16#5000# + 32 beats). The first allocation in a fresh memory_t
  -- starts at address 0, so DUT addresses map 1:1 onto memory addresses.
  constant c_memory_bytes : positive := 32 * 1024;
  constant memory : memory_t := new_memory;
  constant axi_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    address_stall_probability => 0.3,
    data_stall_probability => 0.3,
    min_response_latency => 0 ns,
    max_response_latency => 3 * c_clk_period
  );

  signal m_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal m_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  constant reference_data_queue : queue_t := new_queue;
  signal num_packets_checked : natural := 0;
  signal stream_checker_enable : std_ulogic := '1';

  -- Set (by the request-issuing procedure) to the absolute, monotonically
  -- increasing R-beat index (across the whole testbench run) at which the
  -- RRESP override below should present a non-OKAY RRESP to the DUT. -1
  -- means "never".
  signal inject_error_at_beat : integer := -1;
  signal total_r_beats_served : natural := 0;

  signal num_requests_completed : natural := 0;

  ------------------------------------------------------------------------------
  -- Deterministic word pattern shared between the memory model backing the
  -- AXI slave BFM (RDATA source) and the reference bytes pushed for the
  -- AXI4-Stream checker -- both index by absolute byte address so a
  -- request's stream content is fully predictable regardless of how it got
  -- split into bursts.
  ------------------------------------------------------------------------------
  function word_pattern(byte_addr : natural) return unsigned is
    -- Knuth multiplicative hash constant (0x9E3779B1 = 2654435761). Computed
    -- with explicit unsigned arithmetic and a truncating 32-bit resize
    -- (intentional modulo-2**32 wraparound) rather than 'natural'/'integer'
    -- multiplication, since the literal itself exceeds the 32-bit signed
    -- 'integer' range and would overflow at analysis/run time otherwise.
    constant word_idx : unsigned(31 downto 0) := to_unsigned(byte_addr / c_bytes_per_beat, 32);
    constant multiplier : unsigned(31 downto 0) := x"9e3779b1";
    variable product : unsigned(63 downto 0);
  begin
    product := word_idx * multiplier;
    return product(31 downto 0) + to_unsigned(12345, 32);
  end function;

  procedure push_reference_bytes(base_addr : natural; length_bytes : natural) is
    variable ref : integer_array_t;
    variable word_value : unsigned(31 downto 0);
    variable byte_idx : natural := 0;
  begin
    if length_bytes = 0 then
      return;
    end if;
    ref := new_1d(length => length_bytes, bit_width => 8, is_signed => false);
    for word_offset in 0 to length_bytes / c_bytes_per_beat - 1 loop
      word_value := word_pattern(base_addr + word_offset * c_bytes_per_beat);
      for byte_in_word in 0 to c_bytes_per_beat - 1 loop
        set(
          arr => ref,
          idx => byte_idx,
          value => to_integer(word_value(8 * (byte_in_word + 1) - 1 downto 8 * byte_in_word))
        );
        byte_idx := byte_idx + 1;
      end loop;
    end loop;
    push_ref(reference_data_queue, ref);
  end procedure;

  procedure fill_memory_pattern(base_addr : natural; length_bytes : natural) is
  begin
    for word_offset in 0 to length_bytes / c_bytes_per_beat - 1 loop
      write_word(
        memory => memory,
        address => base_addr + word_offset * c_bytes_per_beat,
        word => std_ulogic_vector(word_pattern(base_addr + word_offset * c_bytes_per_beat))
      );
    end loop;
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;
  test_runner_watchdog(runner, 1 ms);


  ------------------------------------------------------------------------------
  main : process
    variable buf : buffer_t;
    variable num_requests_expected : natural := 0;
    -- A 'length_bytes = 0' request produces no AXI4-Stream packet at all (no
    -- data to send -- see 'cnn_accel_axi_read_dma.vhd's header comment on
    -- the degenerate case), so it must not count towards the packet-level
    -- exit condition below; tracked separately from 'num_requests_expected'.
    variable num_packets_expected : natural := 0;

    procedure do_request(
      addr : natural;
      length_bytes : natural;
      push_reference : boolean := true;
      expect_resp_error : boolean := false
    ) is
    begin
      fill_memory_pattern(base_addr => addr, length_bytes => length_bytes);
      if push_reference then
        push_reference_bytes(base_addr => addr, length_bytes => length_bytes);
      end if;

      if length_bytes > 0 then
        num_packets_expected := num_packets_expected + 1;
      end if;

      req_m2s.req.addr <= to_unsigned(addr, 32);
      req_m2s.req.length <= to_unsigned(length_bytes, 32);
      req_m2s.valid <= '1';
      wait until rising_edge(clk) and req_s2m.ready = '1';
      req_m2s.valid <= '0';

      wait until rising_edge(clk) and dma_done = '1';
      check_equal(resp_error, expect_resp_error, "resp_error for request at addr=" & to_string(addr));

      num_requests_expected := num_requests_expected + 1;
      wait until rising_edge(clk) and num_requests_completed = num_requests_expected;
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    buf := allocate(memory, num_bytes => c_memory_bytes, name => "dma_source");
    check_equal(base_address(buf), 0, "memory region must start at address 0 so DUT addresses map 1:1");

    wait until rising_edge(clk) and reset = '0';

    if run("test_single_short_burst") then
      do_request(addr => 16#0000#, length_bytes => 16 * c_bytes_per_beat);

    elsif run("test_multi_burst_over_256_beats") then
      -- 300 beats: forced to split into two AR bursts (256 + 44) purely by
      -- the 256-beat ARLEN limit -- address 0 keeps the 4 KiB boundary far
      -- away (beat 1024) so it does not also participate in this split.
      do_request(addr => 16#0000#, length_bytes => 300 * c_bytes_per_beat);

    elsif run("test_4k_boundary_split") then
      -- 16 beats starting 8 beats before a 4 KiB boundary: forced to split
      -- into two 8-beat AR bursts purely by the 4 KiB rule (well under the
      -- 256-beat ARLEN limit).
      do_request(addr => 4096 - 8 * c_bytes_per_beat, length_bytes => 16 * c_bytes_per_beat);

    elsif run("test_backpressure") then
      do_request(addr => 16#1000#, length_bytes => 64 * c_bytes_per_beat);

    elsif run("test_resp_error") then
      inject_error_at_beat <= total_r_beats_served + 2;
      wait for 0 ns;
      do_request(addr => 16#2000#, length_bytes => 8 * c_bytes_per_beat, expect_resp_error => true);
      inject_error_at_beat <= -1;

    elsif run("test_reset_mid_transfer") then
      -- Accept a request, then reset before any AR/R beat has even been
      -- issued -- no data has entered any resetless internal FIFO yet, so
      -- there is nothing to leak into the fresh request that follows (see
      -- proposal doc section 4 and the module doc's "Verification notes"
      -- for the scoping of what a mid-transfer reset does and does not
      -- guarantee for the already-resetless hdl-modules submodules).
      fill_memory_pattern(base_addr => 16#3000#, length_bytes => 64 * c_bytes_per_beat);
      req_m2s.req.addr <= to_unsigned(16#3000#, 32);
      req_m2s.req.length <= to_unsigned(64 * c_bytes_per_beat, 32);
      req_m2s.valid <= '1';
      wait until rising_edge(clk) and req_s2m.ready = '1';
      req_m2s.valid <= '0';

      wait for 2 * c_clk_period;
      reset <= '1';
      wait for 4 * c_clk_period;
      reset <= '0';

      wait until rising_edge(clk) and req_s2m.ready = '1';

      do_request(addr => 16#4000#, length_bytes => 32 * c_bytes_per_beat);

    elsif run("test_zero_length") then
      do_request(addr => 16#5000#, length_bytes => 0, push_reference => false);

    end if;

    wait until
      num_requests_completed = num_requests_expected and
      num_packets_checked = num_packets_expected and
      rising_edge(clk);

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  -- dma_done must fire exactly once per accepted request: count pulses
  -- between successive 'req_m2s.valid and req_s2m.ready' accept events.
  ------------------------------------------------------------------------------
  done_once_check : process
    variable dma_done_pulses_this_request : natural := 0;
  begin
    wait until rising_edge(clk);

    if reset = '1' then
      dma_done_pulses_this_request := 0;
    else
      if req_m2s.valid = '1' and req_s2m.ready = '1' then
        check_equal(
          dma_done_pulses_this_request, 0,
          "dma_done must not still be pending from a previous request when a new one is accepted"
        );
      end if;

      if dma_done = '1' then
        dma_done_pulses_this_request := dma_done_pulses_this_request + 1;
        check_equal(dma_done_pulses_this_request, 1, "dma_done pulsed more than once for one request");
        num_requests_completed <= num_requests_completed + 1;
      end if;

      if req_m2s.valid = '1' and req_s2m.ready = '1' then
        dma_done_pulses_this_request := 0;
      end if;
    end if;
  end process;


  ------------------------------------------------------------------------------
  -- AXI4 read slave BFM, backed by the VUnit memory model pre-filled with
  -- 'word_pattern' by each request.
  ------------------------------------------------------------------------------
  axi_read_m2s <= (ar => m_axi_ar_m2s, r => m_axi_r_m2s);
  m_axi_ar_s2m <= axi_read_s2m_bfm.ar;

  axi_read_slave_inst : entity bfm.axi_read_slave
    generic map (
      axi_slave => axi_slave,
      data_width => c_axi_data_width,
      id_width => c_axi_id_width,
      address_width => c_axi_addr_width
    )
    port map (
      clk => clk,
      --
      axi_read_m2s => axi_read_m2s,
      axi_read_s2m => axi_read_s2m_bfm
    );


  ------------------------------------------------------------------------------
  -- Passive RRESP override: forwards the BFM's R channel untouched except
  -- for the resp field on the single beat whose absolute index equals
  -- 'inject_error_at_beat'. 'total_r_beats_served' only advances on a
  -- completed handshake, so the presented resp is stable while valid is
  -- high and not yet accepted.
  ------------------------------------------------------------------------------
  rresp_override : process(all)
  begin
    m_axi_r_s2m <= axi_read_s2m_bfm.r;
    if total_r_beats_served = inject_error_at_beat then
      m_axi_r_s2m.resp <= axi_resp_slverr;
    end if;
  end process;

  r_beat_counter : process
  begin
    wait until rising_edge(clk);
    if m_axi_r_s2m.valid = '1' and m_axi_r_m2s.ready = '1' then
      total_r_beats_served <= total_r_beats_served + 1;
    end if;
  end process;


  ------------------------------------------------------------------------------
  axi_stream_slave_inst : entity bfm.axi_stream_slave
    generic map (
      data_width => c_axi_data_width,
      reference_data_queue => reference_data_queue,
      stall_config => (stall_probability => 0.5, min_stall_cycles => 1, max_stall_cycles => 8)
    )
    port map (
      clk => clk,
      --
      ready => m_stream_s2m.ready,
      valid => m_stream_m2s.valid,
      last => m_stream_m2s.last,
      data => m_stream_m2s.data(c_axi_data_width - 1 downto 0),
      --
      enable => stream_checker_enable,
      num_packets_checked => num_packets_checked
    );


  ------------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_axi_read_dma
    generic map (
      g_axi_addr_width => c_axi_addr_width,
      g_axi_data_width => c_axi_data_width,
      g_axi_id_width => c_axi_id_width
    )
    port map (
      clk => clk,
      reset => reset,
      --
      req_m2s => req_m2s,
      req_s2m => req_s2m,
      --
      dma_done => dma_done,
      resp_error => resp_error,
      --
      m_axi_ar_m2s => m_axi_ar_m2s,
      m_axi_ar_s2m => m_axi_ar_s2m,
      m_axi_r_m2s => m_axi_r_m2s,
      m_axi_r_s2m => m_axi_r_s2m,
      --
      m_stream_m2s => m_stream_m2s,
      m_stream_s2m => m_stream_s2m
    );

end architecture tb;
