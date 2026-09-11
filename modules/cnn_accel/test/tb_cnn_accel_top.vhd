library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
use cnn_accel.cnn_accel_python_ffi_pkg.all;

-- VUnit-5 testbench for cnn_accel_top -- the project's ONE AND ONLY
-- top-level testbench, specified by
-- modules/cnn_accel/doc/cnn_accel_top_v2_arch.md section 11.
--
-- It is deliberately and completely generic. It contains no convolution
-- model, no expected tensor values, no tensor shapes and no opcode
-- knowledge beyond the CSR map of section 8, and it never references a
-- DUT-internal signal (no external names, no hierarchical references, no
-- forcing). Every numerical claim about a run -- and every byte that goes
-- INTO DDR, not just what comes out -- is made in Python, live, over
-- VUnit's Python FFI ('python_call'/'python_execute'). No file is read or
-- written anywhere in this testbench: adding a test adds a Python
-- function (see accel_v2/cases*.py and test/python_bridge/
-- top_level_bridge.py) and never a line of VHDL.
--
-- What it actually does, once, for whichever case 'g_case_name' names:
--
--   1. Model DDR as one flat read_and_write allocation in a 'memory_t',
--      starting at address 0 so DUT addresses are memory addresses.
--   2. Select that case ('top_level_bridge.set_test_case') and fetch its
--      per-run values live (program base address, export/input DDR
--      windows, whether it expects an error).
--   3. Write the compiler's own output (descriptor chain, weight/bias/
--      scale/LUT tables) region by region, then the graph's own input
--      tensors, each via one 'python_call' per region (see
--      'cnn_accel_python_ffi_pkg.vhd's 'ffi_write_indexed_bytes'/
--      'ffi_write_bytes').
--   4. Serve the DUT's single AXI4 master port from that same 'memory_t'
--      via bfm.axi_slave -- the only path to memory the DUT has.
--   5. Drive PROGRAM_BASE_ADDR + CTRL.START over AXI4-Lite (bfm.
--      axi_lite_master) using exclusively the GENERATED register
--      procedures, poll STATUS to DONE/ERROR.
--   6. Read every counter register plus the exported DDR region and hand
--      them straight to 'top_level_bridge.check_result' (one
--      'python_call'), which verifies the run via 'TbCase.check_live'.
--
-- The 'axi_*' counters handed to Python come from the passive monitor
-- process below, which only watches 'm_axi' handshakes. They exist so
-- that Python can cross-check the DUT's self-reported DDR_RD_BYTES /
-- DDR_WR_BYTES against an independent observation: the residency
-- invariants of arch doc section 10 must not be provable solely by the
-- DUT's own bookkeeping.
entity tb_cnn_accel_top is
  generic (
    -- Size of the modelled DDR, bytes. One flat read_and_write allocation
    -- starting at address 0, so DUT addresses map 1:1 onto memory
    -- addresses. Matches the DUT's 'g_ddr_limit' by default.
    g_ddr_bytes : positive := 16#0020_0000#;
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
    -- Which pre-built 'TbCase' this run checks against -- passed to
    -- 'top_level_bridge.set_test_case' once, before CTRL.START. Reads from
    -- 'accel_v2.cases.all_cases()' plus the other six catalogues
    -- 'module_cnn_accel.py' registers configs from. The ONE generic that
    -- has to stay a generic: every other per-case value (the program
    -- base address, the export/input DDR windows, whether the program
    -- is expected to error) is fetched live instead, right after
    -- 'set_test_case' -- the case itself already knows all of it, so
    -- there is no reason for 'module_cnn_accel.py' to also copy it into
    -- a generic (see 'get_program_start_address'/'get_output_region'/
    -- 'get_input_region'/'get_expect_error' in top_level_bridge.py).
    g_case_name : string;
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

  -- Lower case, zero padded, exactly 'num_digits' hex digits. Still used
  -- by 'describe_status' below for diagnostic messages, even though the
  -- old 'result.csv'/'counters.csv' writers that used to be its main
  -- callers are gone.
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
    -- The exported region, read one byte per 'read_word' call and
    -- handed to Python directly (see 'ffi_export_bytes').
    variable export_bytes : integer_array_t;
    -- Discards 'python_call("set_test_case", ...)''s return value: Python's
    -- 'set_test_case' has nothing meaningful to report back, so only the
    -- call (and its exception-to-VHDL-failure path, should the case name
    -- be unknown) matters.
    variable discard : integer;

    -- Per-case values, fetched live from the selected 'TbCase' right
    -- after 'set_test_case' -- the case already knows all of this, so it
    -- is never duplicated into a generic (see top_level_bridge.py's
    -- 'get_program_start_address'/'get_output_region'/'get_input_region'/
    -- 'get_expect_error').
    variable region : integer_array_t;
    variable v_program_base : natural;
    variable v_export_base : natural;
    variable v_export_bytes : natural;
    variable v_inputs_base : natural;
    variable v_inputs_bytes : natural;
    variable v_expect_error : boolean;

    -- The compiler's own output (descriptor chain, weight/bias/scale/LUT
    -- tables), written region by region -- see 'get_program_regions'/
    -- 'get_program_data' in top_level_bridge.py and
    -- 'ffi_write_indexed_bytes' in cnn_accel_python_ffi_pkg.vhd.
    variable compiled_bounds : integer_array_t;
    variable num_compiled_regions : natural;

    -- Every register is read as a raw 'register_t' and, where it has
    -- fields, converted with the generated 'to_cnn_accel_*' function.
    -- Two reasons: the raw value is what 'check_result' is handed,
    -- and a single read keeps the raw value and the decoded fields
    -- consistent (two reads of a live STATUS could disagree).
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

  begin
    test_runner_setup(runner, runner_cfg);

    python_execute(file_name => tb_path(runner_cfg) & "python_bridge/top_level_bridge.py");
    discard := python_call("set_test_case", arg => g_case_name);

    v_program_base := python_call("get_program_start_address");
    v_expect_error := python_call("get_expect_error");
    region := python_call("get_output_region");
    v_export_base := get(region, 0);
    v_export_bytes := get(region, 1);
    region := python_call("get_input_region");
    v_inputs_base := get(region, 0);
    v_inputs_bytes := get(region, 1);

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

    -- The export region is described in whole 64-bit words, so both ends
    -- must be word aligned, and it has to be inside the modelled DDR to
    -- be readable at all.
    check_equal(
      v_export_base mod c_bytes_per_beat,
      0,
      "the case's export_base must be " & to_string(c_bytes_per_beat) & "-byte aligned"
    );
    check_equal(
      v_export_bytes mod c_bytes_per_beat,
      0,
      "the case's export_bytes must be a whole number of " & to_string(c_bytes_per_beat)
      & "-byte words"
    );
    check(
      v_export_base + v_export_bytes <= g_ddr_bytes,
      "the export region must lie inside the modelled DDR (g_ddr_bytes = "
      & to_string(g_ddr_bytes) & ")"
    );

    -- The compiler's own output: one 'ffi_write_indexed_bytes' call per
    -- contiguous compiled region (the descriptor chain, then whichever
    -- of the weight/bias/scale/LUT tables this case actually uses) --
    -- see 'get_program_regions'/'get_program_data' in
    -- top_level_bridge.py for why this is regions rather than one flat
    -- span: 'DdrMap' places each region at a fixed, far-apart base
    -- regardless of how much of it any one case fills.
    compiled_bounds := python_call("get_program_regions");
    num_compiled_regions := length(compiled_bounds) / 2;
    for r in 0 to num_compiled_regions - 1 loop
      ffi_write_indexed_bytes(
        memory, "get_program_data", r,
        base_addr => get(compiled_bounds, 2 * r),
        num_bytes => get(compiled_bounds, 2 * r + 1)
      );
    end loop;
    info(
      "tb_cnn_accel_top: wrote " & to_string(num_compiled_regions)
      & " compiled region(s) via python_call(""get_program_data"")"
    );

    -- The graph's own input tensors: a separate region, written the same
    -- way (see 'get_input_data' in top_level_bridge.py).
    ffi_write_bytes(memory, "get_input_data", v_inputs_base, v_inputs_bytes);
    info(
      "tb_cnn_accel_top: wrote " & to_string(v_inputs_bytes)
      & " input bytes via python_call(""get_input_data"")"
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
      write_cnn_accel_program_base_addr_addr(net, to_unsigned(v_program_base, 32));
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
      -- Every counter register, read once.
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
      -- Every counter, plus the exported DDR region, handed straight to
      -- 'top_level_bridge.check_result' (one python_call), which runs
      -- 'TbCase.check_live' -- no file is read or written anywhere in
      -- this step. No 'check_true' wrapper: 'check_result' raises on
      -- failure, and 'python_call' already reports an uncaught Python
      -- exception to VHDL as a FAILURE with the full traceback, which
      -- says more than any message this call site could write. The
      -- return value is discarded for the same reason it is in
      -- 'set_test_case' above.
      --------------------------------------------------------------------
      export_bytes := ffi_export_bytes(memory, v_export_base, v_export_bytes);

      discard :=
        python_call(
          "check_result",
          arg => export_bytes,
          kwargs =>
            kw("export_base", v_export_base) &
            kw("status", u_unsigned(status_slv)) &
            kw("busy", status.busy) &
            kw("done", status.done) &
            kw("error", status.error) &
            -- Resized to 32 bits like every other kwarg below: passing
            -- 'status.err_code'/'err_pc_low' at their native (4-bit/
            -- 16-bit) field widths corrupted the values on the Python
            -- side -- invisible for every case that only ever exercises
            -- err_code=0/err_pc_low=0, caught by 'err_bad_geometry'
            -- (real, nonzero values) once every case ran the live path.
            kw("err_code", resize(status.err_code, 32)) &
            kw("err_pc_low", resize(status.err_pc_low, 32)) &
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
        );

      --------------------------------------------------------------------
      -- The one and only pass/fail claim this testbench itself makes
      -- about the run. Which ERR_CODE is expected is per-program
      -- knowledge, so it is checked in Python (from the counters just
      -- handed to 'check_result'), not here.
      --------------------------------------------------------------------
      if v_expect_error then
        check_equal(
          status.error,
          '1',
          "the case expects STATUS.ERROR, so the program must end with it set. " & describe_status
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
