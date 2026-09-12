library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.sync_pkg.all;
use vunit_lib.vc_pkg.all;

use work.qspi_pkg.all;
use work.flash_model_pkg.all;

-- QSPI NOR flash verification component.
--
-- This entity is a pin-level shift engine and nothing more. It owns the two
-- things the Python device model cannot have -- the wires and the clock -- and
-- defers every decision about what those wires mean.
--
-- It does not know that 0xEB carries a mode byte, that 0x02 wraps within a
-- page, or that programming is AND-only. After every byte it asks the model
-- "what next?" and receives one packed directive back: receive or transmit,
-- at what lane width, after how many dummy cycles, and which byte to drive.
-- All flash semantics live in python/flash_model/, where they can be
-- unit-tested in milliseconds without a simulator.
--
-- Three processes, deliberately separate:
--
--   * `main` loads the Python bridge, creates the model instance, and then
--     serves the testbench's control messages over com;
--   * `pins` is the shift engine;
--   * `busy_timer` runs the `wait for` of a program or erase. It MUST NOT live
--     in `pins`: the VC would then be deaf to CS and SCK for the whole busy
--     interval, which is exactly when a controller polls the status register.
--     Write-in-progress itself is not a flag here -- the model derives it from
--     a deadline and the `now` this VC passes in -- so there is no flag for
--     this process to race against.
--
-- See doc/flash_model_req.md, doc/flash_model_proposal.md and
-- doc/flash_model_ffi_contract.md.
entity flash_model is
  generic (
    -- Created with new_flash_model. Carries the actor, logger, checker and the
    -- device geometry handed to the Python model.
    g_flash : flash_model_t;
    -- The Python bridge module. Relative names resolve against the directory of
    -- the VUnit run script, so the default works from any testbench without the
    -- testbench having to know where the model lives -- or that it is Python.
    g_python_bridge_path : string := "modules/flash_model/python/flash_model_bridge.py";
    -- Pin-level protocol checking. Off makes the VC ignore SCK period, CS
    -- setup/hold and deselect times entirely; the output delays below still
    -- apply, since those describe what this VC drives rather than what the DUT
    -- does.
    g_protocol_checks : boolean := true
  );
  port (
    m2s : in qspi_m2s_t;
    s2m : out qspi_s2m_t := qspi_s2m_init
  );
end entity;

architecture a of flash_model is

  constant c_logger : logger_t := get_logger(g_flash);
  constant c_checker : checker_t := get_checker(g_flash);

  -- Raised once the Python model exists and its id is in the handle. The pin
  -- engine must not touch the model before this.
  signal initialized : boolean := false;

  -- Set by `pins` to the duration of a program or erase; `busy_timer` turns it
  -- into elapsed simulation time.
  signal busy_request : time := 0 ns;

  -- Two counters rather than one `busy_active` flag, because each signal then
  -- has exactly one driver AND the busy state goes true in the same delta that
  -- `pins` decides it. A single flag set by `busy_timer` would go true one
  -- delta later, leaving a window in which flash_wait_until_ready could sample
  -- it still false and return immediately from a device that is about to be
  -- busy for milliseconds.
  signal busy_started : natural := 0;
  signal busy_finished : natural := 0;

  -- Only drives flash_wait_until_ready. The model's own write-in-progress bit
  -- is derived from its deadline and is the authoritative one; a controller
  -- polling the status register over the bus is always the stronger check.
  signal busy_active : boolean := false;

  -- Output delays, fetched from the model at init: clock-to-output-valid and
  -- CS-high-to-output-Hi-Z. These describe this VC's own driving, so they are
  -- applied with `after`, never asserted on.
  --
  -- Signals rather than shared variables: VHDL-2008 requires a shared variable
  -- to be of a protected type, and a protected type would be overkill for two
  -- values written once at init and read-only thereafter. `pins` waits on
  -- `initialized` before reading them, so it never sees the default.
  signal t_clqv : time := 0 ns;
  signal t_shqz : time := 0 ns;

begin

  ------------------------------------------------------------------------------
  -- Model lifetime and the testbench control surface
  ------------------------------------------------------------------------------

  main : process
    variable v_msg : msg_t;
    variable v_reply : msg_t;
    variable v_msg_type : msg_type_t;
    variable v_id : integer;
    variable v_discard : integer;
    variable v_data : integer_array_t;
    variable v_limits : integer_array_t;
    variable v_address : natural;
    variable v_num_bytes : natural;
    variable v_value : natural;
  begin
    -- The VC loads its own bridge, so a testbench never mentions Python. Cheap
    -- to repeat across instances: python_execute just re-runs the module, and
    -- one interpreter serves the whole simulation.
    python_execute(file_name => g_python_bridge_path);

    -- Fail at time 0 on layout drift rather than as an inexplicable wrong byte
    -- somewhere in the middle of a test.
    check_equal(
      c_checker,
      integer'(python_call("layout_version")), c_layout_version,
      "flash_model: the Python bridge's FFI layout version does not match "
      & "flash_model_pkg's. Regenerate one side from "
      & "doc/flash_model_ffi_contract.md."
    );

    v_id := python_call(
      "flash_create",
      kwargs => kw("profile", to_string(g_flash.p_profile))
        & kw("size_bytes", g_flash.p_size_bytes)
        & kw("page_bytes", g_flash.p_page_bytes)
        & kw("sector_bytes", g_flash.p_sector_bytes)
        & kw("block_bytes", g_flash.p_block_bytes)
        & kw("addr_bytes", g_flash.p_addr_bytes)
        & kw("jedec_id", g_flash.p_jedec_id)
    );
    set_instance_id(g_flash, v_id);

    -- Output delays are VC-side behaviour, so they are cached here once.
    v_limits := python_call("get_timing_limits", kwargs => kw("id", v_id));
    check_equal(
      c_checker, length(v_limits), c_tl_count,
      "flash_model: get_timing_limits returned the wrong number of entries"
    );
    t_clqv <= get(v_limits, c_tl_clqv) * 1 ps;
    t_shqz <= get(v_limits, c_tl_shqz) * 1 ps;
    deallocate(v_limits);

    initialized <= true;

    loop
      receive(net, get_actor(g_flash.p_std_cfg), v_msg);
      v_msg_type := message_type(v_msg);

      if v_msg_type = flash_preload_msg then
        v_address := pop(v_msg);
        v_data := pop_integer_array_t_ref(v_msg);
        v_discard := python_call(
          "preload", arg => v_data,
          kwargs => kw("id", v_id) & kw("addr", v_address)
        );
        -- The procedure handed us a copy precisely so we could own it.
        deallocate(v_data);

      elsif v_msg_type = flash_preload_fill_msg then
        v_address := pop(v_msg);
        v_num_bytes := pop(v_msg);
        v_value := pop(v_msg);
        v_discard := python_call(
          "preload_fill",
          kwargs => kw("id", v_id) & kw("addr", v_address)
            & kw("num_bytes", v_num_bytes) & kw("value", v_value)
        );

      elsif v_msg_type = flash_load_image_msg then
        v_discard := python_call(
          "load_image",
          kwargs => kw("id", v_id) & kw("path", pop_string(v_msg))
            & kw("fmt", pop_string(v_msg)) & kw("base", integer'(pop(v_msg)))
        );

      elsif v_msg_type = flash_read_back_msg then
        v_address := pop(v_msg);
        v_num_bytes := pop(v_msg);
        v_data := python_call(
          "read_back",
          kwargs => kw("id", v_id) & kw("addr", v_address)
            & kw("num_bytes", v_num_bytes)
        );
        v_reply := new_msg(flash_read_back_reply_msg);
        -- Ownership passes to the caller, who deallocates it.
        push_integer_array_t_ref(v_reply, v_data);
        reply(net, v_msg, v_reply);

      elsif v_msg_type = flash_check_content_msg then
        v_address := pop(v_msg);
        v_data := pop_integer_array_t_ref(v_msg);
        -- No check_true around this: a mismatch raises in Python and arrives as
        -- a VUnit failure carrying the address, both values and a traceback.
        v_discard := python_call(
          "check_content", arg => v_data,
          kwargs => kw("id", v_id) & kw("addr", v_address)
        );
        deallocate(v_data);

      elsif v_msg_type = flash_check_content_fill_msg then
        v_address := pop(v_msg);
        v_num_bytes := pop(v_msg);
        v_value := pop(v_msg);
        v_discard := python_call(
          "check_content_fill",
          kwargs => kw("id", v_id) & kw("addr", v_address)
            & kw("num_bytes", v_num_bytes) & kw("value", v_value)
        );

      elsif v_msg_type = flash_written_regions_msg then
        v_data := python_call("written_regions", kwargs => kw("id", v_id));
        v_reply := new_msg(flash_written_regions_reply_msg);
        push_integer_array_t_ref(v_reply, v_data);
        reply(net, v_msg, v_reply);

      elsif v_msg_type = flash_set_timing_enable_msg then
        v_discard := python_call(
          "set_timing_enable",
          kwargs => kw("id", v_id) & kw("enable", boolean'(pop(v_msg)))
        );

      elsif v_msg_type = flash_set_timing_msg then
        v_discard := python_call(
          "set_timing",
          kwargs => kw("id", v_id) & kw("name", pop_string(v_msg))
            & kw("seconds", real(pop_time(v_msg) / 1 ns) * 1.0e-9)
        );

      elsif v_msg_type = flash_set_protection_msg then
        v_address := pop(v_msg);
        v_num_bytes := pop(v_msg);
        v_discard := python_call(
          "set_protection",
          kwargs => kw("id", v_id) & kw("addr", v_address)
            & kw("num_bytes", v_num_bytes) & kw("locked", boolean'(pop(v_msg)))
        );

      elsif v_msg_type = flash_wait_until_ready_msg then
        -- A convenience, not the source of truth: polling the status register
        -- over the bus is what a real controller does and what the tests use.
        -- This still cannot observe a busy period that has not been decided
        -- yet, so a caller that races the command it just issued should poll.
        if busy_active then
          wait until not busy_active;
        end if;
        v_reply := new_msg;
        reply(net, v_msg, v_reply);

      elsif v_msg_type = flash_reset_msg then
        v_discard := python_call("flash_reset", kwargs => kw("id", v_id));
        v_reply := new_msg;
        reply(net, v_msg, v_reply);

      elsif v_msg_type = flash_get_stat_msg then
        v_reply := new_msg(flash_get_stat_reply_msg);
        push(
          v_reply,
          integer'(python_call(
            "get_stat", kwargs => kw("id", v_id) & kw("name", pop_string(v_msg))
          ))
        );
        reply(net, v_msg, v_reply);

      else
        unexpected_msg_type(v_msg_type, g_flash.p_std_cfg);
      end if;

      delete(v_msg);
    end loop;
  end process;

  ------------------------------------------------------------------------------
  -- Busy timer
  ------------------------------------------------------------------------------

  -- Deliberately its own process: see the entity header. This only exists so
  -- flash_wait_until_ready terminates; the model's write-in-progress bit is
  -- derived from a deadline, so nothing here can race a status poll.
  busy_active <= busy_started /= busy_finished;

  busy_timer : process
  begin
    wait until busy_started /= busy_finished;
    wait for busy_request;
    busy_finished <= busy_started;
  end process;

  ------------------------------------------------------------------------------
  -- Shift engine
  ------------------------------------------------------------------------------

  pins : process
    variable v_id : integer;
    variable v_directive : flash_directive_t;
    variable v_byte : std_ulogic_vector(7 downto 0);
    variable v_beats : positive;
    variable v_bits_since_byte : natural;
    variable v_busy_seconds : real;
    variable v_busy_time : time;

    -- Ask the model what to do next. `pass_now` mirrors the previous
    -- directive's volatile flag: a status byte's value depends on simulation
    -- time, an array byte's does not, so the hot path does not pay for a
    -- timestamp it will not use.
    impure function next_directive(byte_in : integer; pass_now : boolean) return flash_directive_t is
    begin
      if pass_now then
        return decode_directive(python_call(
          "xfer",
          kwargs => kw("id", v_id) & kw("byte_in", byte_in) & kw("now_s", now_seconds)
        ));
      end if;
      return decode_directive(python_call(
        "xfer", kwargs => kw("id", v_id) & kw("byte_in", byte_in)
      ));
    end function;

    procedure release_io is
    begin
      s2m.io <= qspi_drive_init after t_shqz;
    end procedure;
  begin
    if not initialized then
      wait until initialized;
    end if;
    v_id := get_instance_id(g_flash);

    loop
      release_io;

      -- Idle until the controller selects the device.
      if m2s.cs_n /= '0' then
        wait until m2s.cs_n = '0';
      end if;

      v_bits_since_byte := 0;
      v_directive := decode_directive(python_call(
        "cs_assert", kwargs => kw("id", v_id) & kw("now_s", now_seconds)
      ));

      -- One iteration per action the model asks for, until it deselects.
      while m2s.cs_n = '0' loop

        -- Dummy cycles are a prefix on this action, not a phase of their own --
        -- which is what makes a lane-width change at the dummy boundary (0x6B)
        -- expressible in a single directive.
        for cycle in 1 to v_directive.pre_dummy_cycles loop
          release_io;
          wait until rising_edge(m2s.sck) or m2s.cs_n /= '0';
          exit when m2s.cs_n /= '0';
        end loop;
        exit when m2s.cs_n /= '0';

        case v_directive.action is

          when receive =>
            release_io;
            v_byte := (others => '0');
            v_beats := qspi_beats_per_byte(v_directive.lanes);
            for beat in 0 to v_beats - 1 loop
              wait until rising_edge(m2s.sck) or m2s.cs_n /= '0';
              exit when m2s.cs_n /= '0';
              v_byte := qspi_byte_insert(
                data => v_byte,
                lanes => v_directive.lanes,
                beat => beat,
                value => qspi_sample_beat(
                  qspi_io_value(m2s, qspi_s2m_init), v_directive.lanes, qspi_master_side
                )
              );
              v_bits_since_byte := (v_bits_since_byte + v_directive.lanes) mod 8;
            end loop;
            exit when m2s.cs_n /= '0';
            v_directive := next_directive(qspi_to_natural(v_byte), v_directive.is_volatile);

          when transmit =>
            v_byte := qspi_to_byte(v_directive.byte_out);
            v_beats := qspi_beats_per_byte(v_directive.lanes);
            for beat in 0 to v_beats - 1 loop
              -- SPI mode 0: the device changes its output on the falling edge
              -- so the controller can sample it on the following rising edge.
              wait until falling_edge(m2s.sck) or m2s.cs_n /= '0';
              exit when m2s.cs_n /= '0';
              s2m.io <= qspi_drive_beat(
                data => v_byte,
                lanes => v_directive.lanes,
                beat => beat,
                driver => qspi_slave_side
              ) after t_clqv;
              wait until rising_edge(m2s.sck) or m2s.cs_n /= '0';
              exit when m2s.cs_n /= '0';
              v_bits_since_byte := (v_bits_since_byte + v_directive.lanes) mod 8;
            end loop;
            exit when m2s.cs_n /= '0';
            -- -1 means "I am clocking out, not in".
            v_directive := next_directive(-1, v_directive.is_volatile);

          when ignore_rest =>
            release_io;
            wait until m2s.cs_n /= '0';

        end case;
      end loop;

      release_io;

      -- Trailing partial byte reporting is not decoration: a real part aborts a
      -- page program or status write whose clock count is not a multiple of 8,
      -- and the model cannot reject a malformed write without knowing this.
      v_busy_seconds := python_call(
        "cs_deassert",
        kwargs => kw("id", v_id) & kw("trailing_bits", v_bits_since_byte)
          & kw("now_s", now_seconds)
      );

      if v_busy_seconds > 0.0 then
        v_busy_time := v_busy_seconds * 1 sec;
        -- Physical-times-real rounding is simulator-dependent, so a duration
        -- shorter than the simulator's resolution can silently truncate to
        -- zero and make a busy operation look instantaneous.
        check(
          c_checker, v_busy_time > 0 ns,
          "flash_model: busy duration " & to_string(v_busy_seconds)
          & " s truncated to 0 -- the simulator's time resolution is too coarse"
        );
        busy_request <= v_busy_time;
        busy_started <= busy_started + 1;
      end if;
    end loop;
  end process;

end architecture;
