library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.sync_pkg.all;

library flash_model;
use flash_model.qspi_pkg.all;
use flash_model.qspi_master_pkg.all;
use flash_model.qspi_flash_cmd_pkg.all;
use flash_model.flash_model_pkg.all;

-- Integration proof for the QSPI NOR flash verification component: a QSPI
-- master VC wired pin-to-pin to the flash VC, exercising the device through
-- real bus traffic rather than through its control surface.
--
-- The control surface is used only to set the device up and to inspect it
-- afterwards, which is the division that matters: every assertion about what
-- the device DOES is made about bytes that actually crossed the wires.
--
-- See doc/flash_model_req.md for what each test corresponds to.
entity tb_flash_model is
  generic (runner_cfg : string);
end entity;

architecture tb of tb_flash_model is

  constant c_page_bytes : positive := 256;
  constant c_sector_bytes : positive := 4096;
  constant c_block_bytes : positive := 65536;
  constant c_jedec_id : natural := 16#EF4018#;

  constant c_master : qspi_master_t := new_qspi_master(sck_period => 20 ns);
  constant c_flash : flash_model_t := new_flash_model(
    profile => "generic_16mib",
    page_bytes => c_page_bytes,
    sector_bytes => c_sector_bytes,
    block_bytes => c_block_bytes,
    jedec_id => c_jedec_id
  );

  signal qspi_m2s : qspi_m2s_t := qspi_m2s_init;
  signal qspi_s2m : qspi_s2m_t := qspi_s2m_init;

  -- A byte array of `length` filled with `first`, `first+1`, ... so a
  -- misordered or off-by-one transfer shows up as a wrong value rather than as
  -- a coincidentally-equal one.
  impure function ramp(length : positive; first : natural := 0) return integer_array_t is
    variable v_result : integer_array_t := new_1d(
      length => length, bit_width => 8, is_signed => false
    );
  begin
    for i in 0 to length - 1 loop
      set(v_result, i, (first + i) mod 256);
    end loop;
    return v_result;
  end function;

  impure function bytes_of(bytes : integer_vector) return integer_array_t is
    variable v_result : integer_array_t := new_1d(
      length => bytes'length, bit_width => 8, is_signed => false
    );
  begin
    for i in 0 to bytes'length - 1 loop
      set(v_result, i, bytes(bytes'low + i));
    end loop;
    return v_result;
  end function;

  procedure check_bytes(
    got : integer_array_t;
    expected : integer_array_t;
    msg : string
  ) is
  begin
    check_equal(length(got), length(expected), msg & ": length");
    for i in 0 to length(expected) - 1 loop
      check_equal(
        get(got, i), get(expected, i),
        msg & ": byte " & to_string(i)
      );
    end loop;
  end procedure;

  -- Poll the status register until write-in-progress clears, over the real bus
  -- -- which is what a controller does, and what proves the VC stays responsive
  -- while it is busy.
  procedure poll_until_ready(
    signal net : inout network_t;
    timeout : delay_length := 10 ms
  ) is
    variable v_status : natural;
    constant c_deadline : time := now + timeout;
  begin
    loop
      qspi_flash_read_status(net, c_master, v_status);
      exit when (v_status mod 2) = 0;  -- bit 0 is WIP
      check(now < c_deadline, "poll_until_ready: timed out with WIP still set");
    end loop;
  end procedure;

begin

  ------------------------------------------------------------------------------
  -- Device under test: the two verification components, wired together
  ------------------------------------------------------------------------------

  qspi_master_inst : entity flash_model.qspi_master
    generic map (
      g_qspi_master => c_master
    )
    port map (
      qspi_m2s => qspi_m2s,
      qspi_s2m => qspi_s2m
    );

  flash_model_inst : entity flash_model.flash_model
    generic map (
      g_flash => c_flash
    )
    port map (
      m2s => qspi_m2s,
      s2m => qspi_s2m
    );

  ------------------------------------------------------------------------------
  -- Tests
  ------------------------------------------------------------------------------

  main : process
    variable v_got : integer_array_t;
    variable v_expected : integer_array_t;
    variable v_status : natural;
    variable v_regions : integer_array_t;
    variable v_count : integer;
    variable v_start : time;
  begin
    test_runner_setup(runner, runner_cfg);

    while test_suite loop
      -- Not optional: one Python interpreter serves the whole simulation and
      -- namespaces are not reset between test cases, so without this the
      -- previous test's array, status bits and mode state leak into this one.
      flash_reset(net, c_flash);
      -- Most tests do not care how long an erase takes; the ones that do turn
      -- this back on themselves.
      flash_set_timing_enable(net, c_flash, false);

      if run("test_read_jedec_id") then
        qspi_flash_read_id(net, c_master, v_got, 3);
        check_equal(get(v_got, 0), 16#EF#, "manufacturer id");
        check_equal(get(v_got, 1), 16#40#, "memory type");
        check_equal(get(v_got, 2), 16#18#, "capacity");
        deallocate(v_got);

      elsif run("test_erased_device_reads_all_ones") then
        -- Nothing has been preloaded, so every byte must read as an erased
        -- cell. This is also the proof that sparse storage has a sane default.
        qspi_flash_read(net, c_master, 16#123456#, 8, v_got);
        v_expected := bytes_of((16#FF#, 16#FF#, 16#FF#, 16#FF#,
                               16#FF#, 16#FF#, 16#FF#, 16#FF#));
        check_bytes(v_got, v_expected, "erased device");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_basic_read_returns_preloaded_data") then
        v_expected := ramp(16, 16#A0#);
        flash_preload(net, c_flash, 16#001000#, v_expected);
        qspi_flash_read(net, c_master, 16#001000#, 16, v_got);
        check_bytes(v_got, v_expected, "0x03 read");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_fast_read_matches_basic_read") then
        -- 0x0B differs from 0x03 only by its dummy cycles, so a mismatch here
        -- is a dummy-cycle bug and nothing else.
        v_expected := ramp(32, 16#10#);
        flash_preload(net, c_flash, 16#002000#, v_expected);
        qspi_flash_fast_read(net, c_master, 16#002000#, 32, v_got);
        check_bytes(v_got, v_expected, "0x0B fast read");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_quad_output_read") then
        -- 0x6B changes lane width AT the dummy boundary: address at x1, then
        -- dummy cycles, then data at x4. This is the case that justifies
        -- pre_dummy_cycles being a prefix on the next action rather than a
        -- phase of its own.
        v_expected := ramp(64, 16#30#);
        flash_preload(net, c_flash, 16#003000#, v_expected);
        qspi_flash_quad_output_read(net, c_master, 16#003000#, 64, v_got);
        check_bytes(v_got, v_expected, "0x6B quad output read");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_quad_io_read") then
        -- 0xEB: opcode at x1, then address AND mode byte at x4, dummy cycles,
        -- then data at x4.
        v_expected := ramp(64, 16#50#);
        flash_preload(net, c_flash, 16#004000#, v_expected);
        qspi_flash_quad_io_read(net, c_master, 16#004000#, 64, v_got);
        check_bytes(v_got, v_expected, "0xEB quad I/O read");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_page_program_then_read_back") then
        v_expected := ramp(c_page_bytes, 16#00#);
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#005000#, v_expected);
        poll_until_ready(net);
        qspi_flash_read(net, c_master, 16#005000#, c_page_bytes, v_got);
        check_bytes(v_got, v_expected, "page program");
        deallocate(v_got);
        deallocate(v_expected);

      elsif run("test_page_program_without_write_enable_is_ignored") then
        v_expected := ramp(8, 16#11#);
        -- No WREN. A real part latches nothing and the array stays erased.
        qspi_flash_page_program(net, c_master, 16#006000#, v_expected);
        poll_until_ready(net);
        flash_check_content_fill(net, c_flash, 16#006000#, 8, 16#FF#);
        flash_get_stat(net, c_flash, "ignored_command_count", v_count);
        check(v_count > 0, "the ignored program should have been counted");
        deallocate(v_expected);

      elsif run("test_page_program_wraps_within_the_page") then
        -- Programming 8 bytes starting 4 before the end of a page must put the
        -- last 4 at the START of the same page, never in the next one.
        v_expected := ramp(8, 16#80#);
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#007000# + c_page_bytes - 4, v_expected);
        poll_until_ready(net);

        qspi_flash_read(net, c_master, 16#007000# + c_page_bytes - 4, 4, v_got);
        check_bytes(v_got, bytes_of((16#80#, 16#81#, 16#82#, 16#83#)), "tail of the page");
        deallocate(v_got);

        qspi_flash_read(net, c_master, 16#007000#, 4, v_got);
        check_bytes(v_got, bytes_of((16#84#, 16#85#, 16#86#, 16#87#)), "wrapped to page start");
        deallocate(v_got);

        -- And the next page must be untouched.
        flash_check_content_fill(net, c_flash, 16#007000# + c_page_bytes, 4, 16#FF#);
        deallocate(v_expected);

      elsif run("test_programming_is_and_only") then
        -- NOR cells only go 1 -> 0 outside an erase: 0xFF & 0xA5 & 0x0F = 0x05.
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#008000#, bytes_of((0 => 16#A5#)));
        poll_until_ready(net);
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#008000#, bytes_of((0 => 16#0F#)));
        poll_until_ready(net);
        flash_check_content(net, c_flash, 16#008000#, bytes_of((0 => 16#05#)));

      elsif run("test_sector_erase") then
        flash_preload_fill(net, c_flash, 16#009000#, c_sector_bytes, 16#00#);
        -- The byte just past the sector must survive, which is what makes this
        -- a granularity test rather than just an erase test.
        flash_preload(net, c_flash, 16#009000# + c_sector_bytes, bytes_of((0 => 16#5A#)));
        qspi_flash_write_enable(net, c_master);
        qspi_flash_sector_erase(net, c_master, 16#009000#);
        poll_until_ready(net);
        flash_check_content_fill(net, c_flash, 16#009000#, c_sector_bytes, 16#FF#);
        flash_check_content(net, c_flash, 16#009000# + c_sector_bytes, bytes_of((0 => 16#5A#)));

      elsif run("test_block_erase") then
        flash_preload_fill(net, c_flash, 16#010000#, c_block_bytes, 16#00#);
        flash_preload(net, c_flash, 16#010000# + c_block_bytes, bytes_of((0 => 16#5A#)));
        qspi_flash_write_enable(net, c_master);
        qspi_flash_block_erase(net, c_master, 16#010000#);
        poll_until_ready(net);
        flash_check_content_fill(net, c_flash, 16#010000#, c_block_bytes, 16#FF#);
        flash_check_content(net, c_flash, 16#010000# + c_block_bytes, bytes_of((0 => 16#5A#)));

      elsif run("test_chip_erase") then
        flash_preload_fill(net, c_flash, 0, 4 * c_sector_bytes, 16#00#);
        flash_preload_fill(net, c_flash, 16#800000#, c_sector_bytes, 16#00#);
        qspi_flash_write_enable(net, c_master);
        qspi_flash_chip_erase(net, c_master);
        poll_until_ready(net);
        flash_check_content_fill(net, c_flash, 0, 4 * c_sector_bytes, 16#FF#);
        flash_check_content_fill(net, c_flash, 16#800000#, c_sector_bytes, 16#FF#);

      elsif run("test_write_enable_latch_is_visible_and_self_clearing") then
        qspi_flash_read_status(net, c_master, v_status);
        check_equal((v_status / 2) mod 2, 0, "WEL starts clear");

        qspi_flash_write_enable(net, c_master);
        qspi_flash_read_status(net, c_master, v_status);
        check_equal((v_status / 2) mod 2, 1, "WREN sets WEL");

        qspi_flash_page_program(net, c_master, 16#00A000#, bytes_of((0 => 16#77#)));
        poll_until_ready(net);
        qspi_flash_read_status(net, c_master, v_status);
        check_equal((v_status / 2) mod 2, 0, "a program clears WEL");

      elsif run("test_protected_region_rejects_program") then
        flash_set_protection(net, c_flash, 16#00B000#, c_sector_bytes, locked => true);
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#00B000#, bytes_of((0 => 16#12#)));
        poll_until_ready(net);
        -- Silently ignored, exactly as a real part behaves: no error, no change.
        flash_check_content_fill(net, c_flash, 16#00B000#, 1, 16#FF#);

      elsif run("test_sparse_preload_leaves_the_gap_erased") then
        flash_preload(net, c_flash, 16#000000#, bytes_of((16#11#, 16#22#)));
        flash_preload(net, c_flash, 16#400000#, bytes_of((16#33#, 16#44#)));
        qspi_flash_read(net, c_master, 16#000000#, 2, v_got);
        check_bytes(v_got, bytes_of((16#11#, 16#22#)), "first region");
        deallocate(v_got);
        qspi_flash_read(net, c_master, 16#400000#, 2, v_got);
        check_bytes(v_got, bytes_of((16#33#, 16#44#)), "second region");
        deallocate(v_got);
        -- The 4 MiB between them was never allocated and must read erased.
        qspi_flash_read(net, c_master, 16#200000#, 4, v_got);
        check_bytes(v_got, bytes_of((16#FF#, 16#FF#, 16#FF#, 16#FF#)), "the gap");
        deallocate(v_got);

      elsif run("test_preload_fill_of_one_mebibyte") then
        -- O(1) in its length: if this ever materializes a megabyte the test
        -- still passes, but the run time will say so loudly.
        flash_preload_fill(net, c_flash, 16#100000#, 1024 * 1024, 16#C3#);
        qspi_flash_read(net, c_master, 16#180000#, 4, v_got);
        check_bytes(v_got, bytes_of((16#C3#, 16#C3#, 16#C3#, 16#C3#)), "middle of the fill");
        deallocate(v_got);
        flash_check_content_fill(net, c_flash, 16#200000#, 4, 16#FF#);

      elsif run("test_written_regions_reports_only_what_was_programmed") then
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#00C000#, ramp(4, 1));
        poll_until_ready(net);
        flash_get_written_regions(net, c_flash, v_regions);
        check_equal(length(v_regions), 2, "exactly one [addr, len] pair");
        check_equal(get(v_regions, 0), 16#00C000#, "region address");
        check_equal(get(v_regions, 1), 4, "region length");
        deallocate(v_regions);

      elsif run("test_four_byte_addressing_reaches_the_same_data") then
        v_expected := ramp(8, 16#C0#);
        flash_preload(net, c_flash, 16#01000000#, v_expected);
        qspi_flash_enter_4byte(net, c_master);
        qspi_flash_read(net, c_master, 16#01000000#, 8, v_got, addr_bytes => 4);
        check_bytes(v_got, v_expected, "4-byte addressed read");
        deallocate(v_got);
        qspi_flash_exit_4byte(net, c_master);
        deallocate(v_expected);

      elsif run("test_qpi_mode_opcode_is_transferred_at_x4") then
        -- In QPI mode the opcode itself is x4, which the VC cannot know in
        -- advance -- it is why cs_assert returns a directive rather than an
        -- acknowledgement.
        v_expected := ramp(8, 16#E0#);
        flash_preload(net, c_flash, 16#00D000#, v_expected);
        qspi_flash_enter_qpi(net, c_master);
        qspi_flash_read(net, c_master, 16#00D000#, 8, v_got,
                        opcode_lanes => 4, lanes => 4);
        check_bytes(v_got, v_expected, "read in QPI mode");
        deallocate(v_got);
        qspi_flash_exit_qpi(net, c_master);
        deallocate(v_expected);

      elsif run("test_erase_takes_its_busy_time") then
        flash_set_timing_enable(net, c_flash, true);
        flash_set_timing(net, c_flash, "tSE", 100 us);
        qspi_flash_write_enable(net, c_master);
        v_start := now;
        qspi_flash_sector_erase(net, c_master, 16#00E000#);
        poll_until_ready(net);
        check(
          now - v_start >= 100 us,
          "the erase reported ready after " & to_string(now - v_start)
          & ", which is less than the 100 us it was configured to take"
        );

      elsif run("test_timing_disabled_makes_erase_instant") then
        flash_set_timing_enable(net, c_flash, false);
        qspi_flash_write_enable(net, c_master);
        v_start := now;
        qspi_flash_sector_erase(net, c_master, 16#00F000#);
        poll_until_ready(net);
        -- Only the bus traffic itself should have taken time.
        check(
          now - v_start < 50 us,
          "with timing disabled the erase still took " & to_string(now - v_start)
        );

      elsif run("test_commands_are_ignored_while_busy") then
        flash_set_timing_enable(net, c_flash, true);
        flash_set_timing(net, c_flash, "tSE", 200 us);
        flash_preload(net, c_flash, 16#011000#, bytes_of((0 => 16#AA#)));

        qspi_flash_write_enable(net, c_master);
        qspi_flash_sector_erase(net, c_master, 16#012000#);

        -- Issued while the erase is still running: a real part drops it, and
        -- crucially the VC must still be answering the bus at all.
        qspi_flash_write_enable(net, c_master);
        qspi_flash_page_program(net, c_master, 16#011000#, bytes_of((0 => 16#00#)));

        poll_until_ready(net);
        flash_check_content(net, c_flash, 16#011000#, bytes_of((0 => 16#AA#)));
      end if;
    end loop;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 50 ms);

end architecture;
