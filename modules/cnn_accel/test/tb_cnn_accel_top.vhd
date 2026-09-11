library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use std.textio.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.memory_pkg.all;
use vunit_lib.axi_slave_pkg.all;
use vunit_lib.python_pkg.all;
use vunit_lib.integer_array_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_lite;
use axi_lite.axi_lite_pkg.all;

library register_file;
use register_file.register_file_pkg.register_t;

library bfm;

library cnn_accel;
use cnn_accel.cnn_accel_regs_pkg.all;
use cnn_accel.cnn_accel_register_record_pkg.all;
use cnn_accel.cnn_accel_register_read_write_pkg.all;

-- VUnit-5 testbench for cnn_accel_top -- the project's ONE AND ONLY
-- top-level testbench, specified by
-- modules/cnn_accel/doc/cnn_accel_top_v2_arch.md section 11.
--
-- It is deliberately and completely generic. It contains no convolution
-- model, no expected tensor values, no tensor shapes and no opcode
-- knowledge beyond the CSR map of section 8, and it never references a
-- DUT-internal signal (no external names, no hierarchical references, no
-- forcing). Every numerical claim about a run is made later, in Python
-- ('post_check'), from the two CSV files written here. All per-test
-- variation arrives as VUnit generics plus files in 'output_path', so
-- adding a test adds a Python function and never a line of VHDL.
--
-- What it actually does, once, for whatever program happens to be in
-- 'mem_image.csv':
--
--   1. Model DDR as one flat read_and_write allocation in a 'memory_t',
--      starting at address 0 so DUT addresses are memory addresses.
--   2. Preload it from '<output_path>/mem_image.csv' (arch doc section 7).
--   3. Serve the DUT's single AXI4 master port from that same 'memory_t'
--      via bfm.axi_slave -- the only path to memory the DUT has.
--   4. Drive PROGRAM_BASE_ADDR + CTRL.START over AXI4-Lite (bfm.
--      axi_lite_master) using exclusively the GENERATED register
--      procedures, poll STATUS to DONE/ERROR, and dump every counter
--      register plus an independent AXI observation to 'counters.csv'.
--   5. Export the requested DDR region to 'result.csv' in the same
--      section-7 format.
--
-- The 'axi_*' rows of 'counters.csv' come from the passive monitor
-- process below, which only watches 'm_axi' handshakes. They exist so
-- that Python can cross-check the DUT's self-reported DDR_RD_BYTES /
-- DDR_WR_BYTES against an independent observation: the residency
-- invariants of arch doc section 10 must not be provable solely by the
-- DUT's own bookkeeping.
entity tb_cnn_accel_top is
  generic (
    -- VUnit's own per-test-config output directory (VUnit fills this in,
    -- with a trailing separator). Every file below is read/written
    -- directly in this directory: 'mem_image.csv' in, 'result.csv' and
    -- 'counters.csv' out.
    output_path : string;
    -- Size of the modelled DDR, bytes. One flat read_and_write allocation
    -- starting at address 0, so DUT addresses map 1:1 onto memory
    -- addresses. Matches the DUT's 'g_ddr_limit' by default.
    g_ddr_bytes : positive := 16#0020_0000#;
    -- Byte address of the program's first descriptor, written to
    -- PROGRAM_BASE_ADDR (arch doc section 6 lays the program at 0x1000).
    g_program_base : natural := 16#0000_1000#;
    -- Byte region exported to 'result.csv' after the run.
    -- 'g_export_bytes' = 0 exports nothing (but still writes a valid
    -- header-only file, see the export step below).
    g_export_base : natural := 16#0010_0000#;
    g_export_bytes : natural := 0;
    -- true when STATUS.ERROR is the expected outcome of this program.
    -- Which ERR_CODE is expected is per-program knowledge that this
    -- testbench must not have -- Python checks that from 'counters.csv'.
    g_expect_error : boolean := false;
    -- Testbench-level liveness bound: cycles waited after START before
    -- the TB itself fails the test. Independent of the DUT's own
    -- watchdog ('g_watchdog_cycles'). The DUT is contractually required
    -- never to hang (arch doc section 9), so hitting this is a real DUT
    -- bug, not a test-tuning knob.
    g_timeout_cycles : positive := 2_000_000;
    -- DUT geometry, forwarded straight to the DUT generics of the same
    -- name. Every other DUT generic is left at its default: they are tied
    -- to the generated constants and to each other by assertions inside
    -- the DUT.
    g_num_banks : positive := 2;
    g_bank_words : positive := 1024;
    g_pe_rows : positive := 8;
    g_ddr_limit : positive := 16#0020_0000#;
    g_watchdog_cycles : positive := 1_000_000;
    -- AXI slave BFM randomized stalling, percent. 0 = no stalling.
    stall_probability_percent : natural := 0;
    -- PILOT (branch feat/vunit-python-ffi-pilot): when true, skip
    -- 'result.csv'/'counters.csv' entirely and instead call
    -- 'top_level_bridge.check_live_result' (a python_call, see
    -- test/python_bridge/top_level_bridge.py) with the same values those
    -- files would have held, right here, before test_runner_cleanup --
    -- 'post_check' is left unset for a config that sets this. Default
    -- false: every existing config's behaviour is completely unchanged.
    g_check_live : boolean := false;
    -- Which pre-built 'TbCase' the live path checks against -- passed to
    -- 'top_level_bridge.select_case' once, before CTRL.START. Only reads
    -- from 'accel_v2.cases.all_cases()' (the pilot's one catalogue), and
    -- only meaningful when 'g_check_live' is true.
    g_case_name : string := "";
    runner_cfg : string
  );
end entity tb_cnn_accel_top;

architecture tb of tb_cnn_accel_top is

  ------------------------------------------------------------------------
  -- Derived AXI geometry. Everything is derived from the DUT's own
  -- 'g_axi_data_width' default -- the generated
  -- 'cnn_accel_constant_max_axi_data_width' -- so the CSV word size, the
  -- BFM data width and the monitor's bytes-per-beat can never disagree
  -- with the DUT.
  ------------------------------------------------------------------------

  constant c_axi_data_width : positive := cnn_accel_constant_max_axi_data_width;
  constant c_bytes_per_beat : positive := c_axi_data_width / 8;
  -- Must match the DUT's 'g_axi_id_width' default, which is left at its
  -- default in the instantiation below. VUnit's AXI slave allocates an
  -- integer_vector_ptr of 2**id_width entries, so 4 is cheap.
  constant c_axi_id_width : natural := 4;

  constant c_clk_period : time := 10 ns;
  -- Modest polling interval so STATUS polling does not dominate
  -- simulation time for long programs, while still resolving DONE within
  -- a microsecond of it happening.
  constant c_poll_interval_cycles : positive := 64;

  ------------------------------------------------------------------------
  -- Clock, reset and the DUT's two interfaces.
  ------------------------------------------------------------------------

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal s_axi_lite_m2s : axi_lite_m2s_t := axi_lite_m2s_init;
  signal s_axi_lite_s2m : axi_lite_s2m_t := axi_lite_s2m_init;

  signal m_axi_m2s : axi_m2s_t := axi_m2s_init;
  signal m_axi_s2m : axi_s2m_t := axi_s2m_init;

  -- Observed but deliberately not checked: IRQ behaviour gets its own
  -- dedicated Python-driven config later, and checking it here would be
  -- an invented requirement.
  signal irq : std_ulogic;

  ------------------------------------------------------------------------
  -- The modelled DDR. One flat allocation is made in the main process
  -- (the first allocation in a fresh 'memory_t' starts at address 0, so
  -- DUT addresses map 1:1 onto memory addresses -- asserted there).
  -- Both BFM halves are built over this SAME 'memory_t', so reads see
  -- what writes wrote.
  ------------------------------------------------------------------------

  constant memory : memory_t := new_memory;

  constant c_stall_probability : real := real(stall_probability_percent) / 100.0;

  -- A response latency only makes sense together with stalling: with
  -- 'stall_probability_percent' = 0 the BFM must be as fast as it can be,
  -- so that the cheap no-stall configs stay cheap.
  function response_latency return time is
  begin
    if stall_probability_percent = 0 then
      return 0 ns;
    end if;
    return 3 * c_clk_period;
  end function;

  constant c_max_response_latency : time := response_latency;

  constant c_axi_read_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    address_stall_probability => c_stall_probability,
    data_stall_probability => c_stall_probability,
    min_response_latency => 0 ns,
    max_response_latency => c_max_response_latency
  );

  constant c_axi_write_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    write_response_fifo_depth => 4,
    address_stall_probability => c_stall_probability,
    data_stall_probability => c_stall_probability
  );

  ------------------------------------------------------------------------
  -- Passive AXI observation (see the entity header). Counted
  -- unconditionally from time zero; never drives anything.
  ------------------------------------------------------------------------

  signal axi_ar_count : natural := 0;
  signal axi_aw_count : natural := 0;
  signal axi_rd_beats : natural := 0;
  signal axi_wr_beats : natural := 0;
  -- Sum over W handshakes of the number of asserted WSTRB bits, i.e. the
  -- number of bytes actually committed to memory.
  signal axi_wr_bytes : natural := 0;
  -- Lowest / highest AWADDR observed, both 0 if no write happened. Kept
  -- as unsigned rather than natural: a 32-bit address does not fit VHDL's
  -- signed 'integer'.
  signal axi_wr_lo_addr : u_unsigned(31 downto 0) := (others => '0');
  signal axi_wr_hi_addr : u_unsigned(31 downto 0) := (others => '0');

  ------------------------------------------------------------------------
  -- Small string/number helpers. The CSV format (arch doc section 7) is
  -- strict on write -- lower case hex, zero padded -- and strict on read,
  -- so the conversions live in one place and are used by both directions.
  ------------------------------------------------------------------------

  -- Decimal rendering of an unsigned of any width. Deliberately not
  -- 'to_string(to_integer(...))': every counter register dumped below is
  -- a full 32-bit unsigned and values at or above 2**31 do not fit VHDL's
  -- signed 'integer'. Long division by 10 has no such limit.
  function to_dec(value : u_unsigned) return string is
    variable rest : u_unsigned(value'length - 1 downto 0) := value;
    -- 'value'length' decimal digits is always enough: 2**n - 1 has at
    -- most ceil(n * log10(2)) + 1 <= n digits for every n >= 1.
    variable digits : string(1 to value'length) := (others => '0');
    variable idx : natural := value'length;
  begin
    if rest = 0 then
      return "0";
    end if;
    while rest /= 0 loop
      digits(idx) := character'val(character'pos('0') + to_integer(rest mod 10));
      idx := idx - 1;
      rest := rest / 10;
    end loop;
    return digits(idx + 1 to digits'high);
  end function;

  function to_dec(value : natural) return string is
  begin
    return to_string(value);
  end function;

  function to_dec(value : std_ulogic) return string is
  begin
    if value = '1' then
      return "1";
    end if;
    return "0";
  end function;

  -- Lower case, zero padded, exactly 'num_digits' hex digits.
  function to_hex(value : u_unsigned; num_digits : positive) return string is
    constant c_nibbles : string(1 to 16) := "0123456789abcdef";
    constant padded : u_unsigned(4 * num_digits - 1 downto 0) := resize(value, 4 * num_digits);
    variable result : string(1 to num_digits);
  begin
    for i in 0 to num_digits - 1 loop
      result(num_digits - i) := c_nibbles(to_integer(padded(4 * i + 3 downto 4 * i)) + 1);
    end loop;
    return result;
  end function;

  function is_blank(c : character) return boolean is
  begin
    -- Space, HT and CR. CR matters: a CRLF-terminated CSV written on
    -- Windows would otherwise make every record 'malformed'.
    return c = ' ' or c = character'val(9) or c = character'val(13);
  end function;

  -- Re-index an arbitrary slice to '1 to n'. Slices keep the bounds they
  -- were cut with, and every parser step below indexes from 1.
  function normalize(s : string) return string is
    variable result : string(1 to s'length) := s;
  begin
    return result;
  end function;

  -- Strip leading/trailing blanks and normalize the index range to
  -- '1 to n', so callers can index the result without worrying about
  -- where the slice came from. 'and' is short-circuit for BOOLEAN in
  -- VHDL, so 's(lo)' is never evaluated out of range.
  function trim(s : string) return string is
    variable lo : integer := s'low;
    variable hi : integer := s'high;
  begin
    while lo <= hi and is_blank(s(lo)) loop
      lo := lo + 1;
    end loop;
    while hi >= lo and is_blank(s(hi)) loop
      hi := hi - 1;
    end loop;
    if lo > hi then
      return "";
    end if;
    return normalize(s(lo to hi));
  end function;

  function to_lower(s : string) return string is
    variable result : string(1 to s'length) := s;
  begin
    for i in result'range loop
      if result(i) >= 'A' and result(i) <= 'Z' then
        result(i) := character'val(character'pos(result(i)) + 32);
      end if;
    end loop;
    return result;
  end function;

  -- 1-based position of the first 'c' in 's', 0 if absent. 's' is always
  -- a 'trim' result here, so its range is '1 to n'.
  function index_of(s : string; c : character) return natural is
  begin
    for i in s'range loop
      if s(i) = c then
        return i;
      end if;
    end loop;
    return 0;
  end function;

  -- -1 for anything that is not a hex digit. Both cases accepted on read.
  function hex_digit_value(c : character) return integer is
  begin
    case c is
      when '0' to '9' => return character'pos(c) - character'pos('0');
      when 'a' to 'f' => return character'pos(c) - character'pos('a') + 10;
      when 'A' to 'F' => return character'pos(c) - character'pos('A') + 10;
      when others => return -1;
    end case;
  end function;

  function is_hex_string(s : string; num_digits : positive) return boolean is
  begin
    if s'length /= num_digits then
      return false;
    end if;
    for i in s'range loop
      if hex_digit_value(s(i)) < 0 then
        return false;
      end if;
    end loop;
    return true;
  end function;

  function hex_to_unsigned(s : string; num_bits : positive) return u_unsigned is
    variable result : u_unsigned(num_bits - 1 downto 0) := (others => '0');
  begin
    for i in s'range loop
      result := shift_left(result, 4);
      result(3 downto 0) := to_unsigned(hex_digit_value(s(i)), 4);
    end loop;
    return result;
  end function;

  procedure write_text_line(file f : text; s : string) is
    variable l : line;
  begin
    write(l, s);
    writeline(f, l);
  end procedure;

  ------------------------------------------------------------------------
  -- CSV memory-image reader (arch doc section 7).
  --
  -- Strictness is the whole point of this parser. A silently misparsed
  -- memory image is indistinguishable from a DUT data bug in the Python
  -- post-check, so every malformed record fails the test immediately and
  -- names the file, the line number and the offending line text. The
  -- first bad record stops the parse: continuing would write garbage
  -- into the memory model and bury the real message under a pile of
  -- follow-on noise.
  ------------------------------------------------------------------------

  procedure load_memory_image(file_name : string; num_words_loaded : out natural) is
    file f : text;
    variable open_status : file_open_status;
    variable l : line;
    variable line_no : natural := 0;
    variable num_records : natural := 0;
    variable header_seen : boolean := false;
    variable stop_parsing : boolean := false;

    procedure bad_line(line_text : string; reason : string) is
    begin
      check_failed(
        "tb_cnn_accel_top: '" & file_name & "' line " & to_string(line_no) & ": " & reason
        & ". Offending line: '" & line_text & "'"
      );
    end procedure;

    -- One physical line of the file. 'abort_parse' is set on the first
    -- malformed record.
    procedure handle_line(raw : string; abort_parse : out boolean) is
      constant txt : string := trim(raw);
      variable comma_pos : natural := 0;
      variable addr_u : u_unsigned(31 downto 0);
      variable data_u : u_unsigned(63 downto 0);
    begin
      abort_parse := false;

      -- Blank lines and lines whose first non-blank character is '#' are
      -- comments.
      if txt'length = 0 or txt(1) = '#' then
        return;
      end if;

      -- The first non-comment line must be the 'address,data' header.
      if not header_seen then
        header_seen := true;
        if to_lower(txt) /= "address,data" then
          bad_line(txt, "expected the required 'address,data' header line");
          abort_parse := true;
        end if;
        return;
      end if;

      comma_pos := index_of(txt, ',');
      if comma_pos = 0 then
        bad_line(txt, "record has no ',' separator");
        abort_parse := true;
        return;
      end if;

      if not is_hex_string(txt(1 to comma_pos - 1), 8) then
        bad_line(txt, "address field must be exactly 8 hex digits");
        abort_parse := true;
        return;
      end if;

      if not is_hex_string(txt(comma_pos + 1 to txt'length), 16) then
        bad_line(txt, "data field must be exactly 16 hex digits");
        abort_parse := true;
        return;
      end if;

      addr_u := hex_to_unsigned(txt(1 to comma_pos - 1), 32);
      data_u := hex_to_unsigned(txt(comma_pos + 1 to txt'length), 64);

      -- Checked as unsigned, before any 'to_integer': an address at or
      -- above 2**31 would overflow VHDL's signed 'integer'.
      if addr_u mod c_bytes_per_beat /= 0 then
        bad_line(
          txt,
          "address is not " & to_string(c_bytes_per_beat) & "-byte aligned"
        );
        abort_parse := true;
        return;
      end if;

      if resize(addr_u, 33) + c_bytes_per_beat > to_unsigned(g_ddr_bytes, 33) then
        bad_line(
          txt,
          "address is outside the modelled DDR (g_ddr_bytes = " & to_string(g_ddr_bytes) & ")"
        );
        abort_parse := true;
        return;
      end if;

      -- Default (little) endianness, so the 16 hex digits are exactly the
      -- 64-bit word as it appears on AXI RDATA/WDATA and 'read_word' with
      -- the same endianness recovers them byte for byte.
      write_word(
        memory => memory,
        address => to_integer(addr_u),
        word => std_logic_vector(data_u)
      );
      num_records := num_records + 1;
    end procedure;

  begin
    num_words_loaded := 0;

    file_open(open_status, f, file_name, read_mode);
    if open_status /= open_ok then
      check_failed(
        "tb_cnn_accel_top: could not open the memory image '" & file_name
        & "'. It is written by the Python pre_config hook before the simulation starts."
      );
      return;
    end if;

    while not endfile(f) loop
      readline(f, l);
      line_no := line_no + 1;
      handle_line(l.all, stop_parsing);
      exit when stop_parsing;
    end loop;

    file_close(f);

    if not stop_parsing and not header_seen then
      check_failed(
        "tb_cnn_accel_top: '" & file_name & "' contains no 'address,data' header line"
      );
    end if;

    num_words_loaded := num_records;
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;

  -- Generously above the testbench's own 'g_timeout_cycles' bound, which
  -- is the check that is supposed to fire (with a diagnosable message) if
  -- the DUT never reaches DONE/ERROR. The extra margin covers reset, the
  -- CSR reads and the file I/O around the run.
  test_runner_watchdog(runner, 2 * g_timeout_cycles * c_clk_period + 1 ms);


  ------------------------------------------------------------------------
  -- DUT. Direct entity instantiation, forwarding only the five geometry
  -- generics this testbench parameterizes; every other generic stays at
  -- its default (they are tied to the generated constants and to each
  -- other by assertions inside the DUT).
  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_top
    generic map (
      g_pe_rows => g_pe_rows,
      g_num_banks => g_num_banks,
      g_bank_words => g_bank_words,
      g_ddr_limit => g_ddr_limit,
      g_watchdog_cycles => g_watchdog_cycles
    )
    port map (
      clk => clk,
      reset => reset,
      --
      s_axi_lite_m2s => s_axi_lite_m2s,
      s_axi_lite_s2m => s_axi_lite_s2m,
      --
      m_axi_m2s => m_axi_m2s,
      m_axi_s2m => m_axi_s2m,
      --
      irq => irq
    );


  ------------------------------------------------------------------------
  -- The host side: VUnit's AXI-Lite master behind hdl-modules' record
  -- wrapper, on the default 'register_bus_master' bus handle -- the same
  -- default the generated 'cnn_accel_register_read_write_pkg' procedures
  -- use, so no bus handle ever has to be passed explicitly below.
  ------------------------------------------------------------------------
  axi_lite_master_inst : entity bfm.axi_lite_master
    port map (
      clk => clk,
      --
      axi_lite_m2s => s_axi_lite_m2s,
      axi_lite_s2m => s_axi_lite_s2m
    );


  ------------------------------------------------------------------------
  -- The DDR side: the combined read+write AXI slave BFM, both halves
  -- backed by the same 'memory_t'. This is the only path to memory the
  -- DUT has (arch doc section 11).
  ------------------------------------------------------------------------
  axi_slave_inst : entity bfm.axi_slave
    generic map (
      axi_read_slave => c_axi_read_slave,
      axi_write_slave => c_axi_write_slave,
      data_width => c_axi_data_width,
      id_width => c_axi_id_width
    )
    port map (
      clk => clk,
      --
      axi_read_m2s => m_axi_m2s.read,
      axi_read_s2m => m_axi_s2m.read,
      --
      axi_write_m2s => m_axi_m2s.write,
      axi_write_s2m => m_axi_s2m.write
    );


  ------------------------------------------------------------------------
  -- Passive AXI monitor. Purely observational: it drives no DUT input and
  -- no BFM signal, it only counts valid+ready handshakes on the DUT's
  -- 'm_axi' from time zero. Its numbers are what lets Python cross-check
  -- the DUT's own DDR_RD_BYTES / DDR_WR_BYTES counters, so the residency
  -- invariants (arch doc section 10) do not rest on the DUT's own
  -- bookkeeping alone.
  ------------------------------------------------------------------------
  axi_monitor : process
    variable ar_count : natural := 0;
    variable aw_count : natural := 0;
    variable rd_beats : natural := 0;
    variable wr_beats : natural := 0;
    variable wr_bytes : natural := 0;
    variable lo_addr : u_unsigned(31 downto 0) := (others => '0');
    variable hi_addr : u_unsigned(31 downto 0) := (others => '0');
    variable aw_addr : u_unsigned(31 downto 0);
  begin
    wait until rising_edge(clk);

    if m_axi_m2s.read.ar.valid = '1' and m_axi_s2m.read.ar.ready = '1' then
      ar_count := ar_count + 1;
    end if;

    if m_axi_s2m.read.r.valid = '1' and m_axi_m2s.read.r.ready = '1' then
      rd_beats := rd_beats + 1;
    end if;

    if m_axi_m2s.write.aw.valid = '1' and m_axi_s2m.write.aw.ready = '1' then
      aw_count := aw_count + 1;
      aw_addr := m_axi_m2s.write.aw.addr(aw_addr'range);
      if aw_count = 1 then
        lo_addr := aw_addr;
        hi_addr := aw_addr;
      else
        if aw_addr < lo_addr then
          lo_addr := aw_addr;
        end if;
        if aw_addr > hi_addr then
          hi_addr := aw_addr;
        end if;
      end if;
    end if;

    if m_axi_m2s.write.w.valid = '1' and m_axi_s2m.write.w.ready = '1' then
      wr_beats := wr_beats + 1;
      -- Only the byte lanes that actually exist at this data width; the
      -- record is sized for the widest AXI this project's axi_pkg allows.
      for byte_lane in 0 to c_bytes_per_beat - 1 loop
        if m_axi_m2s.write.w.strb(byte_lane) = '1' then
          wr_bytes := wr_bytes + 1;
        end if;
      end loop;
    end if;

    axi_ar_count <= ar_count;
    axi_aw_count <= aw_count;
    axi_rd_beats <= rd_beats;
    axi_wr_beats <= wr_beats;
    axi_wr_bytes <= wr_bytes;
    axi_wr_lo_addr <= lo_addr;
    axi_wr_hi_addr <= hi_addr;
  end process;


  ------------------------------------------------------------------------
  -- The one and only test case. All variation is generics + files; there
  -- is deliberately no per-network or per-test-case logic here.
  ------------------------------------------------------------------------
  main : process
    variable ddr : buffer_t;
    variable num_words : natural;
    -- PILOT (g_check_live): the exported region, built the same way
    -- 'result.csv' is (one byte per 'read_word' call), but handed to
    -- Python directly instead of written to a file.
    variable export_bytes : integer_array_t;

    -- Every register is read as a raw 'register_t' and, where it has
    -- fields, converted with the generated 'to_cnn_accel_*' function.
    -- Two reasons: the raw value is what 'counters.csv' reports, and a
    -- single read keeps the raw value and the decoded fields consistent
    -- (two reads of a live STATUS could disagree).
    variable status_slv : register_t := (others => '0');
    variable status : cnn_accel_status_t := cnn_accel_status_init;
    variable hw_info_slv : register_t := (others => '0');
    variable hw_info2_slv : register_t := (others => '0');
    variable hw_info3_slv : register_t := (others => '0');
    variable cmd_count_slv : register_t := (others => '0');
    variable cycle_count_slv : register_t := (others => '0');
    variable compute_cycles_slv : register_t := (others => '0');
    variable stall_cycles_slv : register_t := (others => '0');
    variable ddr_rd_bytes_slv : register_t := (others => '0');
    variable ddr_wr_bytes_slv : register_t := (others => '0');
    variable tensor_load_count_slv : register_t := (others => '0');
    variable tensor_store_count_slv : register_t := (others => '0');
    variable weight_load_bytes_slv : register_t := (others => '0');
    variable local_bytes_slv : register_t := (others => '0');

    variable start_time : time;
    variable elapsed_cycles : natural := 0;

    file f : text;
    variable open_status : file_open_status;

    -- Renders the last STATUS read both raw and field-by-field. Used by
    -- the timeout and the unexpected-error messages, which both have to
    -- be diagnosable straight from the log.
    impure function describe_status return string is
    begin
      return "STATUS=0x" & to_hex(u_unsigned(status_slv), 8)
        & " (busy=" & to_string(status.busy)
        & ", done=" & to_string(status.done)
        & ", error=" & to_string(status.error)
        & ", err_code=" & to_dec(status.err_code)
        & ", err_pc_low=" & to_dec(status.err_pc_low) & ")";
    end function;

    procedure open_output(file_base_name : string) is
    begin
      file_open(open_status, f, output_path & file_base_name, write_mode);
      if open_status /= open_ok then
        check_failed(
          "tb_cnn_accel_top: could not open '" & output_path & file_base_name & "' for writing"
        );
      end if;
    end procedure;

    procedure put_counter(counter_name : string; value : string) is
    begin
      write_text_line(f, counter_name & "," & value);
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    if g_check_live then
      python_execute(file_name => tb_path(runner_cfg) & "python_bridge/top_level_bridge.py");
      python_execute(source => "select_case(" & """" & g_case_name & """" & ")");
    end if;

    ----------------------------------------------------------------------
    -- The modelled DDR. The first allocation in a fresh 'memory_t' starts
    -- at address 0, which is what makes DUT addresses and memory
    -- addresses the same number everywhere in this file and in Python.
    ----------------------------------------------------------------------
    ddr := allocate(
      memory,
      num_bytes => g_ddr_bytes,
      name => "ddr",
      permissions => read_and_write
    );
    check_equal(
      base_address(ddr),
      0,
      "the DDR allocation must start at address 0 so DUT addresses map 1:1 onto memory addresses"
    );

    -- The export region is described in whole 64-bit words by the CSV
    -- format, so both ends must be word aligned, and it has to be inside
    -- the modelled DDR to be readable at all.
    check_equal(
      g_export_base mod c_bytes_per_beat,
      0,
      "g_export_base must be " & to_string(c_bytes_per_beat) & "-byte aligned"
    );
    check_equal(
      g_export_bytes mod c_bytes_per_beat,
      0,
      "g_export_bytes must be a whole number of " & to_string(c_bytes_per_beat) & "-byte words"
    );
    check(
      g_export_base + g_export_bytes <= g_ddr_bytes,
      "the export region must lie inside the modelled DDR (g_ddr_bytes = "
      & to_string(g_ddr_bytes) & ")"
    );

    load_memory_image(output_path & "mem_image.csv", num_words);
    info(
      "tb_cnn_accel_top: loaded " & to_string(num_words) & " words from '"
      & output_path & "mem_image.csv'"
    );

    ----------------------------------------------------------------------
    -- Release reset and let the DUT settle before the first CSR access.
    ----------------------------------------------------------------------
    reset <= '1';
    for i in 1 to 8 loop
      wait until rising_edge(clk);
    end loop;
    reset <= '0';
    for i in 1 to 8 loop
      wait until rising_edge(clk);
    end loop;

    if run("test_run_program") then

      --------------------------------------------------------------------
      -- The entire host-side program-launch sequence: a base address and
      -- a start bit. Everything else the DUT discovers from memory
      -- itself (arch doc section 6). CTRL.START is self-clearing in
      -- hardware, so it is never written back to '0'.
      --------------------------------------------------------------------
      write_cnn_accel_program_base_addr_addr(net, to_unsigned(g_program_base, 32));
      write_cnn_accel_ctrl_start(net, '1');

      --------------------------------------------------------------------
      -- Poll STATUS to DONE or ERROR. The elapsed count is taken from
      -- simulation time rather than from an explicit loop counter, so
      -- the time spent inside the AXI-Lite reads themselves counts
      -- towards the bound too.
      --------------------------------------------------------------------
      start_time := now;
      loop
        read_cnn_accel_status(net, status_slv);
        status := to_cnn_accel_status(status_slv);
        exit when status.done = '1' or status.error = '1';

        elapsed_cycles := (now - start_time) / c_clk_period;
        if elapsed_cycles > g_timeout_cycles then
          check_failed(
            "tb_cnn_accel_top: neither DONE nor ERROR arrived within g_timeout_cycles = "
            & to_string(g_timeout_cycles) & " cycles after CTRL.START. Last " & describe_status
            & ". The DUT is contractually required never to hang (arch doc section 9), so this"
            & " is a DUT bug rather than a test-tuning problem."
          );
          exit;
        end if;

        for i in 1 to c_poll_interval_cycles loop
          wait until rising_edge(clk);
        end loop;
      end loop;

      --------------------------------------------------------------------
      -- Every counter register, read once, in 'counters.csv' order.
      --------------------------------------------------------------------
      read_cnn_accel_hw_info(net, hw_info_slv);
      read_cnn_accel_hw_info2(net, hw_info2_slv);
      read_cnn_accel_hw_info3(net, hw_info3_slv);
      read_cnn_accel_cmd_count(net, cmd_count_slv);
      read_cnn_accel_cycle_count(net, cycle_count_slv);
      read_cnn_accel_compute_cycles(net, compute_cycles_slv);
      read_cnn_accel_stall_cycles(net, stall_cycles_slv);
      read_cnn_accel_ddr_rd_bytes(net, ddr_rd_bytes_slv);
      read_cnn_accel_ddr_wr_bytes(net, ddr_wr_bytes_slv);
      read_cnn_accel_tensor_load_count(net, tensor_load_count_slv);
      read_cnn_accel_tensor_store_count(net, tensor_store_count_slv);
      read_cnn_accel_weight_load_bytes(net, weight_load_bytes_slv);
      read_cnn_accel_local_bytes(net, local_bytes_slv);

      --------------------------------------------------------------------
      -- PILOT (g_check_live): the same counters and the same exported
      -- region, handed straight to 'top_level_bridge.check_live_result'
      -- (one python_call) instead of 'counters.csv'/'result.csv'. See
      -- that function's docstring for the exact argument-to-column
      -- mapping -- it is deliberately the same set, same names, so this
      -- branch and the 'else' branch below are checking identical data,
      -- just by two different transports.
      --------------------------------------------------------------------
      if g_check_live then
        export_bytes := new_1d(length => g_export_bytes, bit_width => 8, is_signed => false);
        for byte_index in 0 to g_export_bytes - 1 loop
          set(
            export_bytes, byte_index,
            to_integer(
              u_unsigned(
                read_word(memory => memory, address => g_export_base + byte_index, bytes_per_word => 1)
              )
            )
          );
        end loop;

        check_true(
          python_call(
            "check_live_result",
            arg => export_bytes,
            kwargs =>
              kw("export_base", g_export_base) &
              kw("status", u_unsigned(status_slv)) &
              kw("busy", status.busy) &
              kw("done", status.done) &
              kw("error", status.error) &
              kw("err_code", std_ulogic_vector(status.err_code)) &
              kw("err_pc_low", std_ulogic_vector(status.err_pc_low)) &
              kw("hw_info", u_unsigned(hw_info_slv)) &
              kw("hw_info2", u_unsigned(hw_info2_slv)) &
              kw("hw_info3", u_unsigned(hw_info3_slv)) &
              kw("cmd_count", u_unsigned(cmd_count_slv)) &
              kw("cycle_count", u_unsigned(cycle_count_slv)) &
              kw("compute_cycles", u_unsigned(compute_cycles_slv)) &
              kw("stall_cycles", u_unsigned(stall_cycles_slv)) &
              kw("ddr_rd_bytes", u_unsigned(ddr_rd_bytes_slv)) &
              kw("ddr_wr_bytes", u_unsigned(ddr_wr_bytes_slv)) &
              kw("tensor_load_count", u_unsigned(tensor_load_count_slv)) &
              kw("tensor_store_count", u_unsigned(tensor_store_count_slv)) &
              kw("weight_load_bytes", u_unsigned(weight_load_bytes_slv)) &
              kw("local_bytes", u_unsigned(local_bytes_slv)) &
              kw("axi_ar_count", axi_ar_count) &
              kw("axi_aw_count", axi_aw_count) &
              kw("axi_rd_beats", axi_rd_beats) &
              kw("axi_wr_beats", axi_wr_beats) &
              kw("axi_wr_bytes", axi_wr_bytes) &
              kw("axi_wr_lo_addr", axi_wr_lo_addr) &
              kw("axi_wr_hi_addr", axi_wr_hi_addr)
          ),
          "check_live_result reported a failure for case " & g_case_name & " -- see the printed "
          & "Python traceback/report above for which check failed"
        );

      else

      --------------------------------------------------------------------
      -- 'counters.csv': the DUT's own bookkeeping first, then this
      -- testbench's independent AXI observation. Python compares the two.
      --------------------------------------------------------------------
      open_output("counters.csv");
      write_text_line(f, "name,value");
      put_counter("status", to_dec(u_unsigned(status_slv)));
      put_counter("busy", to_dec(status.busy));
      put_counter("done", to_dec(status.done));
      put_counter("error", to_dec(status.error));
      put_counter("err_code", to_dec(status.err_code));
      put_counter("err_pc_low", to_dec(status.err_pc_low));
      put_counter("hw_info", to_dec(u_unsigned(hw_info_slv)));
      put_counter("hw_info2", to_dec(u_unsigned(hw_info2_slv)));
      put_counter("hw_info3", to_dec(u_unsigned(hw_info3_slv)));
      put_counter("cmd_count", to_dec(u_unsigned(cmd_count_slv)));
      put_counter("cycle_count", to_dec(u_unsigned(cycle_count_slv)));
      put_counter("compute_cycles", to_dec(u_unsigned(compute_cycles_slv)));
      put_counter("stall_cycles", to_dec(u_unsigned(stall_cycles_slv)));
      put_counter("ddr_rd_bytes", to_dec(u_unsigned(ddr_rd_bytes_slv)));
      put_counter("ddr_wr_bytes", to_dec(u_unsigned(ddr_wr_bytes_slv)));
      put_counter("tensor_load_count", to_dec(u_unsigned(tensor_load_count_slv)));
      put_counter("tensor_store_count", to_dec(u_unsigned(tensor_store_count_slv)));
      put_counter("weight_load_bytes", to_dec(u_unsigned(weight_load_bytes_slv)));
      put_counter("local_bytes", to_dec(u_unsigned(local_bytes_slv)));
      put_counter("axi_ar_count", to_dec(axi_ar_count));
      put_counter("axi_aw_count", to_dec(axi_aw_count));
      put_counter("axi_rd_beats", to_dec(axi_rd_beats));
      put_counter("axi_wr_beats", to_dec(axi_wr_beats));
      put_counter("axi_rd_bytes", to_dec(axi_rd_beats * c_bytes_per_beat));
      put_counter("axi_wr_bytes", to_dec(axi_wr_bytes));
      put_counter("axi_wr_lo_addr", to_dec(axi_wr_lo_addr));
      put_counter("axi_wr_hi_addr", to_dec(axi_wr_hi_addr));
      file_close(f);

      --------------------------------------------------------------------
      -- 'result.csv': the exported region in the section-7 format. Always
      -- written, even when 'g_export_bytes' = 0, so that 'post_check' can
      -- tell "this config exports nothing" from "the testbench died".
      --------------------------------------------------------------------
      open_output("result.csv");
      write_text_line(f, "# cnn_accel memory image v1");
      write_text_line(
        f,
        "# written by tb_cnn_accel_top, export region ["
        & to_dec(to_unsigned(g_export_base, 32)) & ", "
        & to_dec(to_unsigned(g_export_base + g_export_bytes, 32)) & ")"
      );
      write_text_line(f, "address,data");
      if g_export_bytes > 0 then
        for word_index in 0 to g_export_bytes / c_bytes_per_beat - 1 loop
          write_text_line(
            f,
            to_hex(to_unsigned(g_export_base + word_index * c_bytes_per_beat, 32), 8) & ","
            & to_hex(
              u_unsigned(
                read_word(
                  memory => memory,
                  address => g_export_base + word_index * c_bytes_per_beat,
                  bytes_per_word => c_bytes_per_beat
                )
              ),
              2 * c_bytes_per_beat
            )
          );
        end loop;
      end if;
      file_close(f);

      end if; -- g_check_live

      --------------------------------------------------------------------
      -- The one and only pass/fail claim this testbench makes about the
      -- run itself. Which ERR_CODE is expected is per-program knowledge,
      -- so it is checked in Python from 'counters.csv', not here.
      --------------------------------------------------------------------
      if g_expect_error then
        check_equal(
          status.error,
          '1',
          "g_expect_error is true, so the program must end with STATUS.ERROR set. " & describe_status
        );
      else
        check_equal(
          status.error,
          '0',
          "the program must not end with STATUS.ERROR set. " & describe_status
        );
      end if;

    end if;

    test_runner_cleanup(runner);
  end process;

end architecture tb;
