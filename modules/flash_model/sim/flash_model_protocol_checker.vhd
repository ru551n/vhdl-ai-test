library ieee;
use ieee.std_logic_1164.all;

library vunit_lib;
context vunit_lib.vunit_context;

use work.qspi_pkg.all;
use work.flash_model_pkg.all;

-- Pin-level AC protocol checker for the QSPI NOR flash verification component.
--
-- A passive observer of the master-to-slave half of the bus. It drives nothing
-- and it decodes nothing: it measures the intervals between edges of the three
-- pins the controller owns -- SCK, CS_n and the master's IO drive -- and
-- reports every interval that is shorter than the device profile allows.
--
-- Why this is a separate entity rather than a fourth process in flash_model:
-- the shift engine's loop is written around "what does the model want next",
-- and interleaving edge-interval bookkeeping into it would couple two things
-- that fail for completely unrelated reasons. Keeping it out here also means
-- g_enable can remove the whole thing from elaboration, and a negative test can
-- instantiate it on its own against a hand-driven bus.
--
-- The delay/check split. get_timing_limits() returns ten picosecond values, and
-- they are NOT ten of the same kind of thing. Indices 0 .. c_tl_first_delay - 1
-- describe what the *controller* must do: SCK period and high/low time, CS
-- setup and hold around the clock burst, the CS-high gap between commands, and
-- data setup/hold around the sampling edge. Those are the DUT's obligations, so
-- they are asserted here. Indices c_tl_clqv and c_tl_shqz describe what the
-- *device* does -- how long after a clock edge its output is valid, and how
-- long after deselect it lets go of the wires. Those pins are driven by the VC
-- itself, so flash_model schedules them with `after` and this entity must never
-- look at them: a checker that asserted on t_clqv would be asserting that the
-- VC's own output delay had not happened yet, and would fire on every single
-- read beat. Hence the port is `integer_vector(0 to c_tl_first_delay - 1)` --
-- the delays are not merely unused here, they are not even in scope.
--
-- The limits are a port and not a generic because they do not exist at
-- elaboration: flash_model fetches them from the Python profile at time 0, so
-- they arrive one delta into the simulation. `limits_valid` is what says they
-- have; nothing is measured before it rises, which also conveniently skips the
-- reset-value edges of the pins.
--
-- A limit of 0 means "not specified, do not check" -- see
-- doc/flash_model_ffi_contract.md, "Timing limits".
entity flash_model_protocol_checker is
  generic (
    -- The VC's own checker, so that a negative test can mock/unmock the logger
    -- behind it and assert that a violation actually fired.
    g_checker : checker_t;
    -- Off removes the checker from the design entirely.
    g_enable : boolean := true
  );
  port (
    -- The controller-driven half of the bus. Watched, never driven.
    m2s : in qspi_m2s_t;
    -- Low until flash_model has the profile's limits from Python.
    limits_valid : in boolean;
    -- get_timing_limits() indices 0 .. c_tl_first_delay - 1, in picoseconds.
    limits_ps : in integer_vector(0 to c_tl_first_delay - 1)
  );
end entity;

architecture a of flash_model_protocol_checker is

  -- Picoseconds as "20 ns" / "7.519 ns". Worth the twenty lines: `to_string` on
  -- a `time` renders it in the simulator's resolution limit, so the same
  -- violation reads "7519000 fs" under GHDL and something else elsewhere, and
  -- the number a reader wants to compare against the datasheet is in neither.
  function ps_image(value_ps : natural) return string is
    constant c_whole : natural := value_ps / 1000;
    constant c_frac : natural := value_ps mod 1000;
    constant c_digits : string(1 to 3) := (
      1 => character'val(character'pos('0') + c_frac / 100),
      2 => character'val(character'pos('0') + (c_frac / 10) mod 10),
      3 => character'val(character'pos('0') + c_frac mod 10)
    );
  begin
    if c_frac = 0 then
      return to_string(c_whole) & " ns";
    elsif c_frac mod 100 = 0 then
      return to_string(c_whole) & "." & c_digits(1 to 1) & " ns";
    elsif c_frac mod 10 = 0 then
      return to_string(c_whole) & "." & c_digits(1 to 2) & " ns";
    end if;

    return to_string(c_whole) & "." & c_digits & " ns";
  end function;

  -- The one place a violation is reported. The message is built only when the
  -- check fails, which is not just economy: `measured` can legitimately be
  -- seconds (a chip erase between two commands) and seconds do not fit in an
  -- integer number of picoseconds. On the failing path it is by definition
  -- below the limit, so it always does.
  procedure check_min(
    constant measured : in time;
    constant limit_ps : in integer;
    constant what : in string
  ) is
    variable v_measured_ps : natural;
  begin
    -- 0 means "not specified" in the FFI contract; nothing to compare against.
    if limit_ps <= 0 then
      return;
    end if;

    if measured < limit_ps * 1 ps then
      v_measured_ps := measured / 1 ps;
      check_failed(
        g_checker,
        "flash_model protocol: " & what & " " & ps_image(v_measured_ps)
        & " is shorter than the " & ps_image(limit_ps) & " minimum"
      );
    end if;
  end procedure;

begin

  ------------------------------------------------------------------------------
  -- All of it, or none of it
  ------------------------------------------------------------------------------

  enabled_gen : if g_enable generate

    -- One process rather than three, because the interesting checks straddle
    -- two pins: t_slch and t_chsh relate a CS edge to an SCK edge, and the data
    -- setup/hold pair relates an IO change to the sampling edge. Splitting them
    -- would mean duplicating the edge history in each process, or passing it
    -- between processes through signals and paying a delta for it.
    checks : process
      -- SCK edge history. `now` at the last edge of each kind, valid only once
      -- the matching `have` flag is set, so the first edge of a simulation is
      -- not measured against time 0.
      variable v_rise_time : time := 0 fs;
      variable v_fall_time : time := 0 fs;
      variable v_have_rise : boolean := false;
      variable v_have_fall : boolean := false;

      -- CS framing.
      variable v_cs_fall_time : time := 0 fs;
      variable v_cs_rise_time : time := 0 fs;
      variable v_have_cs_rise : boolean := false;
      -- The next SCK rising edge is the first of this command, so it is the one
      -- t_slch is measured to.
      variable v_awaiting_first_sck : boolean := false;
      -- Last SCK edge of either polarity inside the current command, which is
      -- what t_chsh is measured from.
      variable v_sck_edge_time : time := 0 fs;
      variable v_have_sck_edge : boolean := false;

      -- Data-in setup and hold. `v_sample_time` is the last sampling edge at
      -- which the controller was actually driving; a read or dummy beat has no
      -- data-in and so has neither a setup nor a hold obligation.
      variable v_io_change_time : time := 0 fs;
      variable v_sample_time : time := 0 fs;
      variable v_have_sample : boolean := false;

      -- Any lane driven by the controller means this beat carries data in.
      impure function master_driving return boolean is
      begin
        for lane in m2s.io.enable'range loop
          if m2s.io.enable(lane) = '1' then
            return true;
          end if;
        end loop;

        return false;
      end function;
    begin
      if not limits_valid then
        wait until limits_valid;
      end if;

      loop
        wait on m2s;

        ------------------------------------------------------------------------
        -- SCK period, high time and low time
        ------------------------------------------------------------------------

        -- Not gated on CS: a controller that glitches the clock while the
        -- device is deselected is still a controller with a broken clock, and
        -- an interval that spans an idle gap is longer than any minimum, so
        -- leaving the gate out costs nothing in false positives.
        if rising_edge(m2s.sck) then
          if v_have_rise then
            check_min(now - v_rise_time, limits_ps(c_tl_sck_min), "SCK period");
          end if;
          if v_have_fall then
            check_min(now - v_fall_time, limits_ps(c_tl_sck_low_min), "SCK low time");
          end if;
          v_rise_time := now;
          v_have_rise := true;

        elsif falling_edge(m2s.sck) then
          if v_have_rise then
            check_min(now - v_rise_time, limits_ps(c_tl_sck_high_min), "SCK high time");
          end if;
          v_fall_time := now;
          v_have_fall := true;
        end if;

        ------------------------------------------------------------------------
        -- CS framing: t_slch, t_chsh, t_shsl
        ------------------------------------------------------------------------

        if falling_edge(m2s.cs_n) then
          if v_have_cs_rise then
            check_min(
              now - v_cs_rise_time, limits_ps(c_tl_shsl),
              "CS high time between commands"
            );
          end if;
          v_cs_fall_time := now;
          v_awaiting_first_sck := true;
          -- The previous command's clock edges say nothing about this one.
          v_have_sck_edge := false;
          v_have_sample := false;

        elsif rising_edge(m2s.cs_n) then
          if v_have_sck_edge then
            check_min(
              now - v_sck_edge_time, limits_ps(c_tl_chsh),
              "last SCK edge to CS high"
            );
          end if;
          v_cs_rise_time := now;
          v_have_cs_rise := true;
        end if;

        -- Clock edges while selected, for the two checks above. Evaluated after
        -- the CS branch so that an SCK edge coincident with CS falling counts
        -- towards the new command rather than the one that just ended.
        if m2s.cs_n = '0' and (rising_edge(m2s.sck) or falling_edge(m2s.sck)) then
          if v_awaiting_first_sck and rising_edge(m2s.sck) then
            check_min(
              now - v_cs_fall_time, limits_ps(c_tl_slch),
              "CS low to the first SCK rising edge"
            );
            v_awaiting_first_sck := false;
          end if;
          v_sck_edge_time := now;
          v_have_sck_edge := true;
        end if;

        ------------------------------------------------------------------------
        -- Data in, around the sampling edge: t_dvch and t_chdx
        ------------------------------------------------------------------------

        -- SPI mode 0: the device samples what the controller drives on the
        -- rising edge, so the controller owes setup before it and hold after
        -- it. Only for beats it actually drives -- during a dummy or read
        -- phase the IOs are the device's and checking them here would be
        -- checking the VC's own output, exactly what t_clqv must not become.
        if rising_edge(m2s.sck) and m2s.cs_n = '0' and master_driving then
          check_min(
            now - v_io_change_time, limits_ps(c_tl_dvch), "data-in setup"
          );
          v_sample_time := now;
          v_have_sample := true;
        end if;

        if m2s.io'event then
          if v_have_sample and m2s.cs_n = '0' then
            check_min(
              now - v_sample_time, limits_ps(c_tl_chdx), "data-in hold"
            );
          end if;
          -- Including a change into Hi-Z: releasing a lane early is as much a
          -- hold violation as changing its value early.
          v_io_change_time := now;
        end if;
      end loop;
    end process;

  end generate;

end architecture;
