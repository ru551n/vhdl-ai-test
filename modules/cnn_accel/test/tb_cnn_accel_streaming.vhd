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

-- Feature test for the ISA v2.3 streaming-inference interface (spec
-- section 6a, docs/superpowers/specs/2026-09-12-input-output-
-- relocation-and-queueing-design.md): INPUT_ADDR/OUTPUT_ADDR relocation
-- and the one-deep job queue. 'tb_cnn_accel_top' deliberately never
-- exercises either (every one of its configs leaves both registers at
-- their reset value, 0, which is the whole backward-compatibility
-- point) -- this is the dedicated testbench that actually drives them.
--
-- One compiled program ('single_conv', from 'accel_v2.cases') run
-- TWICE without being recompiled, using the exact mechanism the
-- feature is for: the program and its weight/bias/scale/LUT tables are
-- written to DDR ONCE; the input tensor's bytes are written at TWO
-- different DDR locations (the compiled address, and the compiled
-- address plus 'c_reloc_delta'); "job A" runs at INPUT_ADDR=
-- OUTPUT_ADDR=0 (the compiled addresses, unrelocated -- proving the
-- default case is unaffected); "job B" is queued behind it with
-- INPUT_ADDR=OUTPUT_ADDR=c_reloc_delta while job A is still BUSY,
-- proving both relocation and the queue in one sequence: job B's
-- result must come from the RELOCATED addresses, and no second
-- CTRL.START is issued for it -- STATUS.DONE has to assert for it on
-- its own via auto-dispatch.
--
-- A third test then proves the queue really is one-deep: with job A
-- running and job B already queued, a THIRD CTRL.START must be
-- rejected with ERR_QUEUE_FULL, leaving job A and the queued job B
-- both completely unaffected.
entity tb_cnn_accel_streaming is
  generic (
    runner_cfg : string
  );
end entity tb_cnn_accel_streaming;

architecture tb of tb_cnn_accel_streaming is

  constant c_axi_data_width : positive := cnn_accel_constant_max_axi_data_width;
  constant c_axi_id_width : natural := 4;
  constant c_clk_period : time := 10 ns;
  constant c_poll_interval_cycles : positive := 64;
  constant c_timeout_cycles : positive := 200_000;

  -- Added to the compiled input/output addresses for "job B" (and,
  -- negatively, "job C"). Comfortably inside the headroom of both the
  -- INPUTS region (0x80000, 256 KiB before SPILL) and the OUTPUTS
  -- region (0x100000, 1 MiB before LIMIT) for a case as small as
  -- 'single_conv' -- see 'accel_v2.ddrmap.DdrMap'.
  constant c_reloc_delta : natural := 16#1000#;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal s_axi_lite_m2s : axi_lite_m2s_t := axi_lite_m2s_init;
  signal s_axi_lite_s2m : axi_lite_s2m_t := axi_lite_s2m_init;

  signal m_axi_m2s : axi_m2s_t := axi_m2s_init;
  signal m_axi_s2m : axi_s2m_t := axi_s2m_init;

  signal irq : std_ulogic;

  constant memory : memory_t := new_memory;

  constant c_axi_read_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    min_response_latency => 0 ns,
    max_response_latency => 0 ns
  );

  constant c_axi_write_slave : axi_slave_t := new_axi_slave(
    memory => memory,
    address_fifo_depth => 4,
    write_response_fifo_depth => 4
  );

begin

  ------------------------------------------------------------------------
  -- The test. All DDR interaction goes through 'top_level_bridge.py'
  -- (test/python_bridge/top_level_bridge.py) exactly like
  -- 'tb_cnn_accel_top', reusing every one of its 'get_*'/'check_result'
  -- functions unchanged -- this file adds no new Python.
  ------------------------------------------------------------------------
  main : process
    variable ddr : buffer_t;
    variable discard : integer;
    variable region : integer_array_t;
    variable program_addr : natural;
    variable export_base : natural;
    variable export_bytes : natural;
    variable inputs_base : natural;
    variable inputs_bytes : natural;
    variable compiled_bounds : integer_array_t;
    variable num_compiled_regions : natural;
    variable export_data : integer_array_t;

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

    impure function describe_status return string is
    begin
      return "STATUS=" & to_string(unsigned(status_slv))
        & " (busy=" & to_string(status.busy)
        & ", done=" & to_string(status.done)
        & ", error=" & to_string(status.error)
        & ", queued=" & to_string(status.queued)
        & ", err_code=" & to_string(to_integer(status.err_code)) & ")";
    end function;

    -- Poll STATUS until 'done' or 'error' (whichever comes first) since
    -- 'start_time', per 'tb_cnn_accel_top's own watchdog pattern.
    procedure wait_for_done_or_error is
    begin
      start_time := now;
      loop
        read_cnn_accel_status(net, status_slv);
        status := to_cnn_accel_status(status_slv);
        exit when status.done = '1' or status.error = '1';

        elapsed_cycles := (now - start_time) / c_clk_period;
        if elapsed_cycles > c_timeout_cycles then
          check_failed(
            "tb_cnn_accel_streaming: neither DONE nor ERROR arrived within "
            & to_string(c_timeout_cycles) & " cycles. Last " & describe_status
          );
          exit;
        end if;

        for i in 1 to c_poll_interval_cycles loop
          wait until rising_edge(clk);
        end loop;
      end loop;
    end procedure;

    -- Read every counter register and hand them, plus 'export_bytes'
    -- bytes read back from 'base', to 'check_result' -- identical to
    -- 'tb_cnn_accel_top's own final step, just callable per completion
    -- since this testbench observes more than one.
    procedure check_completed_job(
      base : natural;
      bytes : natural;
      -- True for job A's own check in a queued pair: its auto-dispatched
      -- successor starts in the SAME cycle its DONE asserts (spec
      -- section 6a -- no idle gap between frames is the entire point),
      -- so BUSY correctly never reads 0 in between. See
      -- 'TbCase.check_live's own docstring.
      busy_after_done : boolean := false
    ) is
    begin
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

      export_data := ffi_export_bytes(memory, base, bytes);

      discard :=
        python_call(
          "check_result",
          arg => export_data,
          kwargs =>
            kw("export_base", base) &
            kw("status", u_unsigned(status_slv)) &
            kw("busy", status.busy) &
            kw("done", status.done) &
            kw("error", status.error) &
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
            -- No passive AXI monitor in this testbench (see the entity
            -- header: this file only exercises the queue/relocation
            -- protocol, not the residency invariants 'tb_cnn_accel_top'
            -- already covers) -- 0 disables the write-range check
            -- ('TbCase.check_live' only runs it when 'axi_aw_count' > 0).
            kw("axi_aw_count", 0) &
            kw("axi_wr_lo_addr", u_unsigned'(x"00000000")) &
            kw("axi_wr_hi_addr", u_unsigned'(x"00000000")) &
            kw("busy_after_done", busy_after_done)
        );
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    python_execute(file_name => tb_path(runner_cfg) & "python_bridge/top_level_bridge.py");
    discard := python_call("set_test_case", arg => string'("single_conv"));

    program_addr := python_call("get_program_start_address");
    region := python_call("get_output_region");
    export_base := get(region, 0);
    export_bytes := get(region, 1);
    region := python_call("get_input_region");
    inputs_base := get(region, 0);
    inputs_bytes := get(region, 1);

    ----------------------------------------------------------------------
    -- The modelled DDR, and the compiled program -- written once, shared
    -- by both jobs (spec section 6a: the queued job always reruns the
    -- same compiled program).
    ----------------------------------------------------------------------
    ddr := allocate(memory, num_bytes => 16#0020_0000#, name => "ddr", permissions => read_and_write);
    check_equal(base_address(ddr), 0, "the DDR allocation must start at address 0");

    compiled_bounds := python_call("get_program_regions");
    num_compiled_regions := length(compiled_bounds) / 2;
    for r in 0 to num_compiled_regions - 1 loop
      ffi_write_indexed_bytes(
        memory, "get_program_data", r,
        base_addr => get(compiled_bounds, 2 * r),
        num_bytes => get(compiled_bounds, 2 * r + 1)
      );
    end loop;

    -- The graph input's bytes, written at BOTH the compiled address
    -- (job A reads them unrelocated) and the compiled address plus
    -- 'c_reloc_delta' (job B reads them relocated).
    ffi_write_bytes(memory, "get_input_data", inputs_base, inputs_bytes);
    ffi_write_bytes(memory, "get_input_data", inputs_base + c_reloc_delta, inputs_bytes);

    reset <= '1';
    for i in 1 to 8 loop
      wait until rising_edge(clk);
    end loop;
    reset <= '0';
    for i in 1 to 8 loop
      wait until rising_edge(clk);
    end loop;

    write_cnn_accel_program_base_addr_addr(net, to_unsigned(program_addr, 32));

    if run("relocated_and_queued_pair") then

      --------------------------------------------------------------------
      -- Job A: compiled addresses, unrelocated (INPUT_ADDR/OUTPUT_ADDR
      -- are still 0, their reset value -- exactly the case every
      -- 'tb_cnn_accel_top' config already covers, repeated here only so
      -- job B has something real to be queued behind).
      --------------------------------------------------------------------
      write_cnn_accel_ctrl_start(net, '1');

      --------------------------------------------------------------------
      -- Job B: queued while job A is BUSY. The two register writes plus
      -- CTRL.START below take several clock cycles over AXI4-Lite, by
      -- which point job A's STATUS.BUSY is already set -- so this
      -- START is the "queue it" case (spec section 6a), not a second
      -- immediate start.
      --------------------------------------------------------------------
      write_cnn_accel_input_addr_addr(net, to_unsigned(c_reloc_delta, 32));
      write_cnn_accel_output_addr_addr(net, to_unsigned(c_reloc_delta, 32));
      write_cnn_accel_ctrl_start(net, '1');

      read_cnn_accel_status(net, status_slv);
      status := to_cnn_accel_status(status_slv);
      check_equal(status.queued, '1', "job B should be queued. " & describe_status);
      check_equal(status.error, '0', "queueing job B must not itself raise an error");

      -- Job A completes first (it was already running); check it
      -- against the UNRELOCATED (compiled) output address. Its own
      -- auto-dispatched successor (job B) starts in the same cycle its
      -- DONE asserts, so BUSY correctly never reads 0 in between.
      wait_for_done_or_error;
      check_equal(status.error, '0', "job A must not error. " & describe_status);
      check_completed_job(export_base, export_bytes, busy_after_done => true);

      -- Clear DONE (write-1-to-clear) so the NEXT assertion of it can
      -- only be job B's own completion, not a stale read of job A's.
      write_cnn_accel_status_done(net, '1');

      -- Job B auto-dispatches with no second CTRL.START; check it
      -- against the RELOCATED output address.
      wait_for_done_or_error;
      check_equal(status.error, '0', "job B must not error. " & describe_status);
      check_completed_job(export_base + c_reloc_delta, export_bytes);

    elsif run("queue_full_rejected") then

      write_cnn_accel_ctrl_start(net, '1');

      write_cnn_accel_input_addr_addr(net, to_unsigned(c_reloc_delta, 32));
      write_cnn_accel_output_addr_addr(net, to_unsigned(c_reloc_delta, 32));
      write_cnn_accel_ctrl_start(net, '1');

      read_cnn_accel_status(net, status_slv);
      status := to_cnn_accel_status(status_slv);
      check_equal(status.queued, '1', "job B should be queued. " & describe_status);
      check_equal(status.error, '0', "queueing job B must not itself raise an error");

      -- A third START, with the one-deep queue already full: must be
      -- rejected (ERR_QUEUE_FULL), and must not disturb job A (still
      -- running) or job B (still queued).
      write_cnn_accel_input_addr_addr(net, to_unsigned(2 * c_reloc_delta, 32));
      write_cnn_accel_output_addr_addr(net, to_unsigned(2 * c_reloc_delta, 32));
      write_cnn_accel_ctrl_start(net, '1');

      read_cnn_accel_status(net, status_slv);
      status := to_cnn_accel_status(status_slv);
      check_equal(
        status.error, '1',
        "a third START with the queue already full must raise ERR_QUEUE_FULL. "
        & describe_status
      );
      check_equal(to_integer(status.err_code), 16#A#, "expected ERR_QUEUE_FULL (0xA)");
      check_equal(
        status.queued, '1',
        "the rejected third START must not disturb the already-queued job B"
      );

      -- Clear the rejection's ERROR (write-1-to-clear) so it cannot be
      -- mistaken for a real fault below, then let job A and the queued
      -- job B run to completion exactly as in the positive test above,
      -- proving neither was corrupted by the rejected write.
      write_cnn_accel_status_error(net, '1');

      wait_for_done_or_error;
      check_equal(status.error, '0', "job A must not error. " & describe_status);
      check_completed_job(export_base, export_bytes, busy_after_done => true);

      write_cnn_accel_status_done(net, '1');

      wait_for_done_or_error;
      check_equal(status.error, '0', "job B must not error. " & describe_status);
      check_completed_job(export_base + c_reloc_delta, export_bytes);

    end if;

    test_runner_cleanup(runner);
  end process;

  clk <= not clk after c_clk_period / 2;
  test_runner_watchdog(runner, 2 * c_timeout_cycles * c_clk_period + 1 ms);

  dut : entity cnn_accel.cnn_accel_top
    port map (
      clk => clk,
      reset => reset,
      s_axi_lite_m2s => s_axi_lite_m2s,
      s_axi_lite_s2m => s_axi_lite_s2m,
      m_axi_m2s => m_axi_m2s,
      m_axi_s2m => m_axi_s2m,
      irq => irq
    );

  axi_lite_master_inst : entity bfm.axi_lite_master
    port map (
      clk => clk,
      axi_lite_m2s => s_axi_lite_m2s,
      axi_lite_s2m => s_axi_lite_s2m
    );

  axi_slave_inst : entity bfm.axi_slave
    generic map (
      axi_read_slave => c_axi_read_slave,
      axi_write_slave => c_axi_write_slave,
      data_width => c_axi_data_width,
      id_width => c_axi_id_width
    )
    port map (
      clk => clk,
      axi_read_m2s => m_axi_m2s.read,
      axi_read_s2m => m_axi_s2m.read,
      axi_write_m2s => m_axi_m2s.write,
      axi_write_s2m => m_axi_s2m.write
    );

end architecture tb;
