library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.sync_pkg.all;
use vunit_lib.vc_pkg.all;

library flash_model;
use flash_model.qspi_master_pkg.all;
use flash_model.qspi_pkg.all;

-- Protocol-generic QSPI master verification component.
--
-- Drives the clock, the chip select and the master half of the four IO wires
-- for the transactions queued through qspi_master_pkg, and knows nothing
-- beyond that: no opcodes, no address widths, no flash state. Which bytes go
-- out, on how many lanes, with how many dummy cycles in between and how many
-- bytes come back is entirely the caller's business -- see qspi_master_pkg for
-- the transaction shape and qspi_flash_cmd_pkg for the JEDEC layer built on
-- top of it.
--
-- SPI mode 0 (CPOL = 0, CPHA = 0). Within a transaction each SCK cycle is
-- driven as
--
--   apply the master's drive -> wait half a period -> SCK high (both ends
--   sample here) -> wait half a period -> SCK low
--
-- so master outputs change on the falling edge and are stable for a full half
-- period before the rising edge on which the far end samples them. A read beat
-- is the same cycle with the master's output enables cleared, sampling the
-- resolved bus immediately before driving SCK high; the far end is expected to
-- have driven the beat on the preceding falling edge. Dummy cycles are read
-- beats whose sample is discarded, so the master is tri-stated throughout
-- them, which is what makes a lane turnaround between an x1 address phase and
-- an x4 data phase safe.
--
-- Chip select framing belongs to the VC: CS falls half a period before the
-- first rising edge, rises half a period after the last falling edge, and
-- stays high for a full period before the next transaction may start.
entity qspi_master is
  generic (
    g_qspi_master : qspi_master_t
  );
  port (
    qspi_m2s : out qspi_m2s_t := qspi_m2s_init;
    qspi_s2m : in qspi_s2m_t
  );
end entity;

architecture a of qspi_master is

  constant c_actor : actor_t := get_actor(g_qspi_master);
  constant c_logger : logger_t := get_logger(g_qspi_master);
  constant c_checker : checker_t := get_checker(g_qspi_master);

  -- Driven by 'main' and mirrored onto the port, so that the process can read
  -- back its own drive when resolving the bus during a read beat.
  signal m2s : qspi_m2s_t := qspi_m2s_init;

begin

  ------------------------------------------------------------------------------
  qspi_m2s <= m2s;


  ------------------------------------------------------------------------------
  main : process

    variable v_sck_period : delay_length := sck_period(g_qspi_master);

    -- One SCK cycle. 'sample' is the resolved bus immediately before the
    -- rising edge, which is what the far end presented for this beat.
    procedure sck_cycle(variable sample : out qspi_io_t) is
    begin
      wait for v_sck_period / 2;
      sample := qspi_io_value(m2s, qspi_s2m);
      m2s.sck <= '1';
      wait for v_sck_period / 2;
      m2s.sck <= '0';
    end procedure;

    procedure write_phase(
      constant bytes : in integer_array_t;
      constant lanes : in lane_count_t;
      constant phase_name : in string
    ) is
      variable v_byte : std_ulogic_vector(7 downto 0);
      variable v_sample : qspi_io_t;
    begin
      if is_null(bytes) or length(bytes) = 0 then
        return;
      end if;

      debug(
        c_logger,
        "Sending " & integer'image(length(bytes)) & " " & phase_name & " byte(s) on "
          & integer'image(lanes) & " lane(s)"
      );

      for index in 0 to length(bytes) - 1 loop
        v_byte := qspi_to_byte(get(bytes, index));
        for beat in 0 to qspi_beats_per_byte(lanes) - 1 loop
          m2s.io <= qspi_drive_beat(v_byte, lanes, beat, qspi_master_side);
          sck_cycle(v_sample);
        end loop;
      end loop;
    end procedure;

    procedure dummy_phase(constant cycles : in natural) is
      variable v_sample : qspi_io_t;
    begin
      if cycles = 0 then
        return;
      end if;

      debug(c_logger, "Clocking " & integer'image(cycles) & " dummy cycle(s)");

      -- Tri-state before the first dummy cycle, i.e. on the falling edge that
      -- ends the last write beat, so the whole dummy phase is a turnaround.
      m2s.io.enable <= (others => '0');
      for cycle in 1 to cycles loop
        sck_cycle(v_sample);
      end loop;
    end procedure;

    procedure read_phase(
      constant bytes : in integer_array_t;
      constant lanes : in lane_count_t
    ) is
      variable v_byte : std_ulogic_vector(7 downto 0);
      variable v_sample : qspi_io_t;
      variable v_slice : std_ulogic_vector(lanes - 1 downto 0);
    begin
      if length(bytes) = 0 then
        return;
      end if;

      debug(
        c_logger,
        "Receiving " & integer'image(length(bytes)) & " byte(s) on "
          & integer'image(lanes) & " lane(s)"
      );

      m2s.io.enable <= (others => '0');

      for index in 0 to length(bytes) - 1 loop
        v_byte := (others => '0');
        for beat in 0 to qspi_beats_per_byte(lanes) - 1 loop
          sck_cycle(v_sample);
          v_slice := qspi_sample_beat(v_sample, lanes, qspi_slave_side);
          check_false(
            c_checker,
            is_x(v_slice),
            "Read byte " & integer'image(index) & " beat " & integer'image(beat)
              & ": the far end drove " & to_string(v_slice) & " on the data lanes"
          );
          v_byte := qspi_byte_insert(v_byte, lanes, beat, to_x01(v_slice));
        end loop;
        set(bytes, index, qspi_to_natural(v_byte));
      end loop;
    end procedure;

    -- One complete transaction, CS framing included.
    procedure run_transfer(
      constant cmd : in integer_array_t;
      constant cmd_lanes : in lane_count_t;
      constant addr : in integer_array_t;
      constant addr_lanes : in lane_count_t;
      constant wr_data : in integer_array_t;
      constant wr_lanes : in lane_count_t;
      constant dummy_cycles : in natural;
      constant rd_data : in integer_array_t;
      constant read_lanes : in lane_count_t
    ) is
    begin
      m2s.cs_n <= '0';

      write_phase(cmd, cmd_lanes, "command");
      write_phase(addr, addr_lanes, "address");
      write_phase(wr_data, wr_lanes, "write-data");
      dummy_phase(dummy_cycles);
      read_phase(rd_data, read_lanes);

      -- Last falling edge to CS high, then the mandatory CS-high gap before
      -- the next transaction.
      m2s.io.enable <= (others => '0');
      wait for v_sck_period / 2;
      m2s.cs_n <= '1';
      wait for v_sck_period;
    end procedure;

    -- Pop one byte phase, in the order qspi_master_pkg pushed it.
    procedure pop_byte_phase(
      constant msg : in msg_t;
      variable bytes : out integer_array_t;
      variable lanes : out lane_count_t
    ) is
      variable v_length : natural;
      variable v_lanes : lane_count_t;
      variable v_bytes : integer_array_t;
    begin
      v_length := pop_integer(msg);
      v_lanes := pop_integer(msg);
      v_bytes := new_1d(length => v_length, bit_width => 8, is_signed => false);
      for index in 0 to v_length - 1 loop
        set(v_bytes, index, pop_integer(msg));
      end loop;

      bytes := v_bytes;
      lanes := v_lanes;
    end procedure;

    variable msg, reply_msg : msg_t;
    variable msg_type : msg_type_t;

    variable v_cmd, v_addr, v_wr_data, v_rd_data : integer_array_t;
    variable v_cmd_lanes, v_addr_lanes, v_wr_lanes, v_read_lanes : lane_count_t;
    variable v_dummy_cycles, v_num_read_bytes : natural;

  begin
    receive(net, c_actor, msg);
    msg_type := message_type(msg);

    handle_sync_message(net, msg_type, msg);

    if msg_type = qspi_transfer_msg then
      pop_byte_phase(msg, v_cmd, v_cmd_lanes);
      pop_byte_phase(msg, v_addr, v_addr_lanes);
      pop_byte_phase(msg, v_wr_data, v_wr_lanes);
      v_dummy_cycles := pop_integer(msg);
      v_num_read_bytes := pop_integer(msg);
      v_read_lanes := pop_integer(msg);

      v_rd_data := new_1d(length => v_num_read_bytes, bit_width => 8, is_signed => false);

      run_transfer(
        cmd => v_cmd,
        cmd_lanes => v_cmd_lanes,
        addr => v_addr,
        addr_lanes => v_addr_lanes,
        wr_data => v_wr_data,
        wr_lanes => v_wr_lanes,
        dummy_cycles => v_dummy_cycles,
        rd_data => v_rd_data,
        read_lanes => v_read_lanes
      );

      deallocate(v_cmd);
      deallocate(v_addr);
      deallocate(v_wr_data);

      -- Ownership of the read data moves to the caller, which redeems it with
      -- await_qspi_transfer_reply.
      reply_msg := new_msg(qspi_transfer_reply_msg);
      push_ref(reply_msg, v_rd_data);
      reply(net, msg, reply_msg);

    elsif msg_type = qspi_master_set_sck_period_msg then
      v_sck_period := pop_time(msg);
      debug(c_logger, "SCK period set to " & to_string(v_sck_period));
      acknowledge(net, msg, true);

    else
      unexpected_msg_type(msg_type, g_qspi_master.p_std_cfg);
    end if;
  end process;

end architecture;
