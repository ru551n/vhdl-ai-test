library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.queue_pkg.all;
use vunit_lib.sync_pkg.all;

library flash_model;
use flash_model.qspi_flash_cmd_pkg.all;
use flash_model.qspi_master_pkg.all;
use flash_model.qspi_pkg.all;

-- VUnit-5 testbench for the QSPI master verification component.
--
-- The far end here is a deliberately dumb stub slave written inside this file,
-- not the real flash device model: the point is to prove that the master puts
-- the right bits on the right wires at the right times, which a model that
-- also decides what those bits mean would only obscure. The stub is told the
-- shape of each transaction up front through 'slave_cfg' (how many bytes at
-- how many lanes in each phase, how many dummy cycles, how many bytes to send
-- back), counts SCK edges to stay in lockstep, pushes every byte it receives
-- into 'rx_queue' and drives back whatever the test put into 'tx_queue'.
-- Because the stub counts edges rather than decoding a command, a master that
-- emitted the wrong number of dummy cycles would slide the whole read phase
-- and corrupt the data -- which is exactly the check we want.
--
-- Bit-order claims are not checked through the same helper functions the
-- master uses, which would be circular. A separate recorder process samples
-- the resolved four-wire bus and the master's own output enables at every
-- rising edge while CS is low, and the bit-order tests compare that raw trace
-- against sequences worked out by hand from the byte values.
--
-- Covered: MSB-first byte serialization at x1, x2 and x4; the x1 MOSI/MISO
-- lane split; dummy cycles (exact count, master tri-stated throughout); CS
-- framing (SCK low at every CS edge, IOs released while CS is high, a CS-high
-- gap between transactions); read-data capture at every lane width; mixed
-- lane widths within one transaction; the run-time SCK-period setter; and
-- sync_pkg's wait_until_idle against a batch of queued transactions.
entity tb_qspi_master is
  generic (
    runner_cfg : string;
    -- SCK period, in nanoseconds. Every timing expectation below is derived
    -- from this, so the whole suite runs unchanged at any bus speed.
    g_sck_period_ns : positive := 20
  );
end entity;

architecture tb of tb_qspi_master is

  constant c_sck_period : delay_length := g_sck_period_ns * 1 ns;

  constant c_qspi_master : qspi_master_t := new_qspi_master(sck_period => c_sck_period);

  signal qspi_m2s : qspi_m2s_t := qspi_m2s_init;
  signal qspi_s2m : qspi_s2m_t := qspi_s2m_init;

  -- The resolved four wires, as a probe on the board would see them.
  signal io : qspi_io_t := (others => 'Z');

  -- Shape of the transaction the stub slave should expect next. Set by the
  -- test process before the transaction is issued.
  type slave_cfg_t is record
    cmd_bytes : natural;
    cmd_lanes : lane_count_t;
    addr_bytes : natural;
    addr_lanes : lane_count_t;
    wr_bytes : natural;
    wr_lanes : lane_count_t;
    dummy_cycles : natural;
    rd_bytes : natural;
    rd_lanes : lane_count_t;
  end record;

  constant c_slave_cfg_init : slave_cfg_t := (
    cmd_bytes => 0,
    cmd_lanes => 1,
    addr_bytes => 0,
    addr_lanes => 1,
    wr_bytes => 0,
    wr_lanes => 1,
    dummy_cycles => 0,
    rd_bytes => 0,
    rd_lanes => 1
  );

  signal slave_cfg : slave_cfg_t := c_slave_cfg_init;

  -- Bytes the stub slave received, and bytes it should send back.
  constant c_rx_queue : queue_t := new_queue;
  constant c_tx_queue : queue_t := new_queue;

  -- Raw bus trace, one entry per queue per rising SCK edge while CS is low.
  constant c_io_queue : queue_t := new_queue;
  constant c_oe_queue : queue_t := new_queue;

  signal cs_assert_count : natural := 0;

  impure function to_byte_array(values : integer_vector) return integer_array_t is
    variable v_result : integer_array_t := new_1d(
      length => values'length, bit_width => 8, is_signed => false
    );
  begin
    for index in 0 to values'length - 1 loop
      set(v_result, index, values(values'low + index));
    end loop;

    return v_result;
  end function;

begin

  ------------------------------------------------------------------------------
  io <= qspi_io_value(qspi_m2s, qspi_s2m);


  ------------------------------------------------------------------------------
  -- Raw bus trace. Independent of every qspi_pkg helper, so the bit-order
  -- tests below are a real check rather than a tautology.
  record_bus : process
  begin
    wait until rising_edge(qspi_m2s.sck);

    if qspi_m2s.cs_n = '0' then
      push_integer(c_io_queue, qspi_to_natural(io));
      push_integer(c_oe_queue, qspi_to_natural(qspi_m2s.io.enable));
    end if;
  end process;


  ------------------------------------------------------------------------------
  -- Chip-select framing, checked continuously rather than per test.
  check_cs_framing : process
    variable v_last_rise : delay_length := 0 fs;
  begin
    wait on qspi_m2s.cs_n;

    check(qspi_m2s.sck = '0', "SCK must be low whenever CS changes");

    if qspi_m2s.cs_n = '0' then
      cs_assert_count <= cs_assert_count + 1;
      if v_last_rise /= 0 fs then
        check(
          now - v_last_rise >= c_sck_period,
          "CS must stay high for at least one SCK period between transactions"
        );
      end if;
    else
      check_equal(
        qspi_to_natural(qspi_m2s.io.enable), 0, "Master must release the IOs when CS goes high"
      );
      v_last_rise := now;
    end if;
  end process;


  ------------------------------------------------------------------------------
  -- Stub slave. Counts SCK edges through the phase shape it was given.
  stub_slave : process
    variable v_cfg : slave_cfg_t;
    variable v_byte : std_ulogic_vector(7 downto 0);

    procedure receive_phase(constant num_bytes : in natural; constant lanes : in lane_count_t) is
    begin
      for index in 0 to num_bytes - 1 loop
        v_byte := (others => '0');
        for beat in 0 to qspi_beats_per_byte(lanes) - 1 loop
          wait until rising_edge(qspi_m2s.sck);
          v_byte := qspi_byte_insert(
            v_byte, lanes, beat, qspi_sample_beat(io, lanes, qspi_master_side)
          );
        end loop;
        push_integer(c_rx_queue, qspi_to_natural(v_byte));
      end loop;
    end procedure;

  begin
    qspi_s2m.io <= qspi_drive_init;

    wait until falling_edge(qspi_m2s.cs_n);
    v_cfg := slave_cfg;

    receive_phase(v_cfg.cmd_bytes, v_cfg.cmd_lanes);
    receive_phase(v_cfg.addr_bytes, v_cfg.addr_lanes);
    receive_phase(v_cfg.wr_bytes, v_cfg.wr_lanes);

    for cycle in 1 to v_cfg.dummy_cycles loop
      wait until rising_edge(qspi_m2s.sck);
      check_equal(
        qspi_to_natural(qspi_m2s.io.enable),
        0,
        "Master must tri-state every IO during a dummy cycle"
      );
    end loop;

    -- Drive each read beat on the falling edge before the rising edge the
    -- master samples it on, as a real device clocking out on CPOL=0 does.
    for index in 0 to v_cfg.rd_bytes - 1 loop
      v_byte := qspi_to_byte(pop_integer(c_tx_queue));
      for beat in 0 to qspi_beats_per_byte(v_cfg.rd_lanes) - 1 loop
        wait until falling_edge(qspi_m2s.sck);
        qspi_s2m.io <= qspi_drive_beat(v_byte, v_cfg.rd_lanes, beat, qspi_slave_side);
      end loop;
    end loop;

    wait until rising_edge(qspi_m2s.cs_n);
    qspi_s2m.io <= qspi_drive_init;
  end process;


  ------------------------------------------------------------------------------
  qspi_master_inst : entity flash_model.qspi_master
    generic map (
      g_qspi_master => c_qspi_master
    )
    port map (
      qspi_m2s => qspi_m2s,
      qspi_s2m => qspi_s2m
    );


  ------------------------------------------------------------------------------
  main : process

    -- Tell the stub slave what the next transaction looks like.
    procedure configure_slave(
      constant cmd_bytes : in natural := 0;
      constant cmd_lanes : in lane_count_t := 1;
      constant addr_bytes : in natural := 0;
      constant addr_lanes : in lane_count_t := 1;
      constant wr_bytes : in natural := 0;
      constant wr_lanes : in lane_count_t := 1;
      constant dummy_cycles : in natural := 0;
      constant rd_bytes : in natural := 0;
      constant rd_lanes : in lane_count_t := 1
    ) is
    begin
      slave_cfg <= (
        cmd_bytes => cmd_bytes,
        cmd_lanes => cmd_lanes,
        addr_bytes => addr_bytes,
        addr_lanes => addr_lanes,
        wr_bytes => wr_bytes,
        wr_lanes => wr_lanes,
        dummy_cycles => dummy_cycles,
        rd_bytes => rd_bytes,
        rd_lanes => rd_lanes
      );
      -- Let the stub see it before the VC pulls CS low.
      wait for 0 ns;
    end procedure;

    procedure load_tx(constant values : in integer_vector) is
    begin
      for index in values'range loop
        push_integer(c_tx_queue, values(index));
      end loop;
    end procedure;

    -- queue_pkg's length() counts encoded bytes rather than pushed items, so
    -- everything below counts by draining instead.
    procedure drain_trace(variable count : out natural) is
      variable v_count : natural := 0;
      variable v_ignored : integer;
    begin
      while not is_empty(c_io_queue) loop
        v_ignored := pop_integer(c_io_queue);
        v_ignored := pop_integer(c_oe_queue);
        v_count := v_count + 1;
      end loop;

      count := v_count;
    end procedure;

    procedure check_trace_length(
      constant expected_cycles : in natural;
      constant context_msg : in string
    ) is
      variable v_count : natural;
    begin
      drain_trace(v_count);
      check_equal(
        v_count, expected_cycles, context_msg & ": number of SCK cycles while CS was low"
      );
    end procedure;

    -- Compare the recorded raw bus trace against a hand-derived sequence.
    procedure check_trace(
      constant expected_io : in integer_vector;
      constant expected_oe : in integer_vector;
      constant context_msg : in string
    ) is
      variable v_extra : natural;
    begin
      for index in 0 to expected_io'length - 1 loop
        check_false(
          is_empty(c_io_queue),
          context_msg & ": only " & integer'image(index) & " SCK cycles recorded, expected "
            & integer'image(expected_io'length)
        );
        check_equal(
          pop_integer(c_io_queue),
          expected_io(expected_io'low + index),
          context_msg & ": bus value at SCK cycle " & integer'image(index)
        );
        check_equal(
          pop_integer(c_oe_queue),
          expected_oe(expected_oe'low + index),
          context_msg & ": master output enables at SCK cycle " & integer'image(index)
        );
      end loop;

      drain_trace(v_extra);
      check_equal(
        v_extra,
        0,
        context_msg & ": SCK cycles beyond the expected " & integer'image(expected_io'length)
      );
    end procedure;

    procedure check_received(
      constant expected : in integer_vector;
      constant context_msg : in string
    ) is
      variable v_extra : natural := 0;
      variable v_ignored : integer;
    begin
      for index in 0 to expected'length - 1 loop
        check_false(
          is_empty(c_rx_queue),
          context_msg & ": only " & integer'image(index) & " bytes received, expected "
            & integer'image(expected'length)
        );
        check_equal(
          pop_integer(c_rx_queue),
          expected(expected'low + index),
          context_msg & ": byte " & integer'image(index)
        );
      end loop;

      while not is_empty(c_rx_queue) loop
        v_ignored := pop_integer(c_rx_queue);
        v_extra := v_extra + 1;
      end loop;
      check_equal(
        v_extra,
        0,
        context_msg & ": bytes received beyond the expected " & integer'image(expected'length)
      );
    end procedure;

    procedure check_read_data(
      constant data : in integer_array_t;
      constant expected : in integer_vector;
      constant context_msg : in string
    ) is
    begin
      check_equal(length(data), expected'length, context_msg & ": number of bytes read");

      for index in 0 to expected'length - 1 loop
        check_equal(
          get(data, index),
          expected(expected'low + index),
          context_msg & ": read byte " & integer'image(index)
        );
      end loop;
    end procedure;

    -- 0xB2 = 1011_0010 and 0x1F = 0001_1111, the two bytes every bit-order
    -- test sends. Neither is a palindrome under bit reversal (0xB2 reverses to
    -- 0x4D, 0x1F to 0xF8), so an LSB-first master cannot pass by accident --
    -- which 0xA5 and 0x3C, tempting as they look, would have let it do.
    constant c_pattern : integer_vector(0 to 1) := (16#B2#, 16#1F#);

    variable v_cmd, v_addr, v_data, v_read : integer_array_t := null_integer_array;
    variable v_reference : qspi_transfer_reference_t;
    variable v_references : msg_vec_t(0 to 2);
    variable v_status : natural;
    variable v_timestamp : delay_length;

  begin
    test_runner_setup(runner, runner_cfg);

    if run("test_command_phase_x1_is_msb_first") then
      configure_slave(cmd_bytes => 2, cmd_lanes => 1);
      v_cmd := to_byte_array(c_pattern);
      qspi_transfer(net => net, qspi_master => c_qspi_master, cmd => v_cmd, cmd_lanes => 1);

      -- One bit per cycle on IO0, most significant first. Only IO0 is driven,
      -- so the other three wires are Hi-Z and read back as 0.
      check_trace(
        expected_io => (1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1),
        expected_oe => (
          2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#,
          2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#
        ),
        context_msg => "x1 command phase"
      );
      check_received(c_pattern, "x1 command phase");

    elsif run("test_command_phase_x2_is_msb_first") then
      configure_slave(cmd_bytes => 2, cmd_lanes => 2);
      v_cmd := to_byte_array(c_pattern);
      qspi_transfer(net => net, qspi_master => c_qspi_master, cmd => v_cmd, cmd_lanes => 2);

      -- Two bits per cycle on IO1:IO0, most significant pair first:
      -- 0xB2 -> 10 11 00 10, 0x1F -> 00 01 11 11.
      check_trace(
        expected_io => (2#10#, 2#11#, 2#00#, 2#10#, 2#00#, 2#01#, 2#11#, 2#11#),
        expected_oe => (
          2#0011#, 2#0011#, 2#0011#, 2#0011#, 2#0011#, 2#0011#, 2#0011#, 2#0011#
        ),
        context_msg => "x2 command phase"
      );
      check_received(c_pattern, "x2 command phase");

    elsif run("test_command_phase_x4_is_msb_first") then
      configure_slave(cmd_bytes => 2, cmd_lanes => 4);
      v_cmd := to_byte_array(c_pattern);
      qspi_transfer(net => net, qspi_master => c_qspi_master, cmd => v_cmd, cmd_lanes => 4);

      -- One nibble per cycle on IO3:IO0, high nibble first.
      check_trace(
        expected_io => (2#1011#, 2#0010#, 2#0001#, 2#1111#),
        expected_oe => (2#1111#, 2#1111#, 2#1111#, 2#1111#),
        context_msg => "x4 command phase"
      );
      check_received(c_pattern, "x4 command phase");

    elsif run("test_dummy_cycles_are_counted_and_hi_z") then
      -- The shape of a quad output read: opcode at x1, eight dummy cycles,
      -- two bytes at x4. The stub slave counts exactly eight dummy edges, so
      -- a wrong count would shift the read phase and corrupt the data.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, dummy_cycles => 8, rd_bytes => 2, rd_lanes => 4);
      load_tx((16#5A#, 16#C3#));
      v_cmd := to_byte_array((0 => 16#6B#));
      qspi_transfer(
        net => net,
        qspi_master => c_qspi_master,
        cmd => v_cmd,
        data => v_read,
        cmd_lanes => 1,
        dummy_cycles => 8,
        num_read_bytes => 2,
        read_lanes => 4
      );

      -- 8 opcode cycles + 8 dummy + 4 read = 20, and the master drives only
      -- during the opcode: everything after it is Hi-Z on the master side.
      check_trace(
        expected_io => (
          0, 1, 1, 0, 1, 0, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0,
          2#0101#, 2#1010#, 2#1100#, 2#0011#
        ),
        expected_oe => (
          2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0
        ),
        context_msg => "dummy cycles"
      );
      check_received((0 => 16#6B#), "dummy cycles");
      check_read_data(v_read, (16#5A#, 16#C3#), "dummy cycles");

    elsif run("test_read_data_x1_uses_miso") then
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, rd_bytes => 2, rd_lanes => 1);
      load_tx((16#5A#, 16#C3#));
      v_cmd := to_byte_array((0 => 16#03#));
      qspi_transfer(
        net => net,
        qspi_master => c_qspi_master,
        cmd => v_cmd,
        data => v_read,
        cmd_lanes => 1,
        num_read_bytes => 2,
        read_lanes => 1
      );

      check_read_data(v_read, (16#5A#, 16#C3#), "x1 read");

      -- 0x03 on IO0, then 0x5A and 0xC3 on IO1 -- a single-lane read comes
      -- back on MISO, not on the wire the opcode went out on.
      check_trace(
        expected_io => (
          0, 0, 0, 0, 0, 0, 1, 1,
          0, 2, 0, 2, 2, 0, 2, 0,
          2, 2, 0, 0, 0, 0, 2, 2
        ),
        expected_oe => (
          2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#, 2#0001#,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0
        ),
        context_msg => "x1 read"
      );

    elsif run("test_read_data_x2") then
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, dummy_cycles => 4, rd_bytes => 2, rd_lanes => 2);
      load_tx((16#5A#, 16#C3#));
      v_cmd := to_byte_array((0 => 16#3B#));
      qspi_transfer(
        net => net,
        qspi_master => c_qspi_master,
        cmd => v_cmd,
        data => v_read,
        cmd_lanes => 1,
        dummy_cycles => 4,
        num_read_bytes => 2,
        read_lanes => 2
      );

      check_read_data(v_read, (16#5A#, 16#C3#), "x2 read");

    elsif run("test_read_data_x4") then
      configure_slave(cmd_bytes => 1, cmd_lanes => 4, rd_bytes => 4, rd_lanes => 4);
      load_tx((16#00#, 16#FF#, 16#5A#, 16#C3#));
      v_cmd := to_byte_array((0 => 16#0B#));
      qspi_transfer(
        net => net,
        qspi_master => c_qspi_master,
        cmd => v_cmd,
        data => v_read,
        cmd_lanes => 4,
        num_read_bytes => 4,
        read_lanes => 4
      );

      check_read_data(v_read, (16#00#, 16#FF#, 16#5A#, 16#C3#), "x4 read");
      check_received((0 => 16#0B#), "x4 read");

    elsif run("test_mixed_lane_phases") then
      -- A 0xEB quad IO read: opcode at x1, then address plus mode byte at x4,
      -- four dummy cycles, then data at x4. Three lane widths in one frame.
      configure_slave(
        cmd_bytes => 1,
        cmd_lanes => 1,
        addr_bytes => 4,
        addr_lanes => 4,
        dummy_cycles => 4,
        rd_bytes => 3,
        rd_lanes => 4
      );
      load_tx((16#11#, 16#22#, 16#33#));

      qspi_flash_quad_io_read(
        net => net,
        qspi_master => c_qspi_master,
        addr => 16#123456#,
        num_bytes => 3,
        data => v_read,
        addr_bytes => 3,
        dummy_cycles => 4,
        mode_byte => 16#00#
      );

      check_received(
        (16#EB#, 16#12#, 16#34#, 16#56#, 16#00#), "quad IO read"
      );
      check_read_data(v_read, (16#11#, 16#22#, 16#33#), "quad IO read");

      -- 8 opcode + 8 (four bytes at x4) + 4 dummy + 6 data = 26 cycles.
      check_trace_length(26, "quad IO read");

    elsif run("test_cs_framing") then
      configure_slave(cmd_bytes => 1, cmd_lanes => 1);
      v_cmd := to_byte_array((0 => 16#06#));
      qspi_transfer(net => net, qspi_master => c_qspi_master, cmd => v_cmd, cmd_lanes => 1);

      check(qspi_m2s.cs_n = '1', "CS must be high between transactions");
      check_equal(qspi_to_natural(qspi_m2s.io.enable), 0, "IOs must be released between transactions");
      check_equal(cs_assert_count, 1, "exactly one CS assertion so far");

      configure_slave(cmd_bytes => 1, cmd_lanes => 1);
      qspi_transfer(net => net, qspi_master => c_qspi_master, cmd => v_cmd, cmd_lanes => 1);

      check_equal(cs_assert_count, 2, "each transaction gets its own CS assertion");
      check_received((16#06#, 16#06#), "CS framing");

    elsif run("test_wait_until_idle") then
      configure_slave(cmd_bytes => 1, cmd_lanes => 1);

      -- Queue three transactions without waiting for any of them, then prove
      -- wait_until_idle does not return until the bus has actually run them.
      for index in v_references'range loop
        v_cmd := to_byte_array((0 => 16#06# + index));
        qspi_transfer(
          net => net,
          qspi_master => c_qspi_master,
          cmd => v_cmd,
          reference => v_references(index),
          cmd_lanes => 1
        );
        deallocate(v_cmd);
      end loop;

      wait_until_idle(net, as_sync(c_qspi_master));

      check_equal(cs_assert_count, 3, "wait_until_idle returned before the queue drained");
      check_received((16#06#, 16#07#, 16#08#), "wait_until_idle");

      for index in v_references'range loop
        await_qspi_transfer_reply(net, v_references(index));
      end loop;

    elsif run("test_set_sck_period") then
      set_sck_period(net, c_qspi_master, 2 * c_sck_period);

      configure_slave(cmd_bytes => 1, cmd_lanes => 1);
      v_cmd := to_byte_array((0 => 16#9F#));
      qspi_transfer(
        net => net,
        qspi_master => c_qspi_master,
        cmd => v_cmd,
        reference => v_reference,
        cmd_lanes => 1
      );

      wait until rising_edge(qspi_m2s.sck);
      v_timestamp := now;
      wait until rising_edge(qspi_m2s.sck);
      check_equal(now - v_timestamp, 2 * c_sck_period, "SCK period after set_sck_period");

      await_qspi_transfer_reply(net, v_reference);
      check_received((0 => 16#9F#), "set_sck_period");

    elsif run("test_flash_command_layer") then
      -- 0x9F, three ID bytes at x1.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, rd_bytes => 3, rd_lanes => 1);
      load_tx((16#EF#, 16#40#, 16#18#));
      qspi_flash_read_id(net, c_qspi_master, v_read, num_bytes => 3);
      check_received((0 => 16#9F#), "read id");
      check_read_data(v_read, (16#EF#, 16#40#, 16#18#), "read id");

      -- 0x03 with a four-byte address, proving 4-byte addressing mode.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, addr_bytes => 4, addr_lanes => 1,
                      rd_bytes => 1, rd_lanes => 1);
      load_tx((0 => 16#77#));
      qspi_flash_read(
        net => net,
        qspi_master => c_qspi_master,
        addr => 16#01234567#,
        num_bytes => 1,
        data => v_read,
        addr_bytes => 4
      );
      check_received((16#03#, 16#01#, 16#23#, 16#45#, 16#67#), "4-byte read");
      check_read_data(v_read, (0 => 16#77#), "4-byte read");

      -- 0x06 then 0x02 with a payload, the ordinary program sequence.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1);
      qspi_flash_write_enable(net, c_qspi_master);
      check_received((0 => 16#06#), "write enable");

      configure_slave(cmd_bytes => 1, cmd_lanes => 1, addr_bytes => 3, addr_lanes => 1,
                      wr_bytes => 3, wr_lanes => 1);
      v_data := to_byte_array((16#DE#, 16#AD#, 16#BE#));
      qspi_flash_page_program(
        net => net,
        qspi_master => c_qspi_master,
        addr => 16#00A000#,
        data => v_data,
        addr_bytes => 3
      );
      deallocate(v_data);
      check_received(
        (16#02#, 16#00#, 16#A0#, 16#00#, 16#DE#, 16#AD#, 16#BE#), "page program"
      );

      -- 0x05, one status byte.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, rd_bytes => 1, rd_lanes => 1);
      load_tx((0 => 16#02#));
      qspi_flash_read_status(net, c_qspi_master, v_status);
      check_received((0 => 16#05#), "read status");
      check_equal(v_status, 16#02#, "read status value");

      -- 0x20, an erase with only an address.
      configure_slave(cmd_bytes => 1, cmd_lanes => 1, addr_bytes => 3, addr_lanes => 1);
      qspi_flash_sector_erase(net, c_qspi_master, addr => 16#010000#);
      check_received((16#20#, 16#01#, 16#00#, 16#00#), "sector erase");

      -- 0xFF on four lanes, the one command issued from inside QPI mode.
      configure_slave(cmd_bytes => 1, cmd_lanes => 4);
      qspi_flash_exit_qpi(net, c_qspi_master);
      check_received((0 => 16#FF#), "exit QPI");
    end if;

    if not is_null(v_cmd) then
      deallocate(v_cmd);
    end if;
    if not is_null(v_addr) then
      deallocate(v_addr);
    end if;
    if not is_null(v_read) then
      deallocate(v_read);
    end if;

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  test_runner_watchdog(runner, 1 ms);

end architecture;
