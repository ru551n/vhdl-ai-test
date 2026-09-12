library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_v2_pkg.all;
use cnn_accel.cnn_accel_regs_pkg.all;
use cnn_accel.cnn_accel_register_record_pkg.all;

library axi_lite;
use axi_lite.axi_lite_pkg.all;

-- AXI4-Lite control/status register block (doc/cnn_accel_top_v2_arch.md
-- section 8). Direct-instantiates the `hdl-registers`-generated
-- 'cnn_accel_register_file_axi_lite' (per AGENTS.md: cnn_accel's registers
-- are generated from 'module_cnn_accel.py's 'registers_hook()', never
-- hand-declared here) and drives/consumes its typed
-- 'cnn_accel_regs_up_t'/'cnn_accel_regs_down_t'/'cnn_accel_reg_was_written_t'
-- records; this entity only owns the glue logic around the register file --
-- pulse generation, sticky-bit tracking, PROGRAM_BASE_ADDR latching and IRQ
-- masking.
--
-- Register-mode deviation from a plain r/r_w split (documented in
-- 'registers_hook()'s own docstring, module_cnn_accel.py -- repeated here
-- only to the extent it constrains this entity's RTL): 'CTRL' and 'STATUS'
-- both use 'r_wpulse', which reads from 'regs_up' (hardware-computed) and
-- presents a write as a one-cycle pulse in 'regs_down' plus a
-- 'reg_was_written' strobe, rather than storing it. That pulse is exactly
-- CTRL's self-clearing START/ABORT signal, but for STATUS it is only the
-- *candidate* write-1-to-clear event -- 'hdl-registers'/
-- 'axi_lite_register_file' do not reduce "write 1 clears a sticky bit" for
-- us, so this entity ANDs the one-cycle 'regs_down.status.done'/'.error'
-- pulse against 'reg_was_written.status' itself (see 'status_tracking'
-- below), and a same-cycle 'seq_done'/'seq_error' is defined to win over a
-- same-cycle clear so a real completion/error event is never lost under a
-- racing W1C.
entity cnn_accel_csr is
  generic (
    g_pe_rows : positive;
    g_pe_cols : positive;
    g_tile_channels : positive;
    g_max_kernel_size : positive;
    -- Pooling's own, separate kernel bound (cnn_accel_top's
    -- 'g_max_pool_kernel_size'). Reported to the host via
    -- HW_INFO3.MAX_POOL_KERNEL_SIZE -- distinct from HW_INFO.MAX_KERNEL_SIZE.
    g_max_pool_kernel_size : positive;
    -- Elaborated per-row activation tile depth (cnn_accel_top's
    -- 'g_max_row_tile_words'). Reported to the host via
    -- HW_INFO3.MAX_ROW_TILE_WORDS.
    g_max_row_tile_words : positive;
    -- Size of the local tensor scratchpad (cnn_accel_tensor_mem), bytes.
    -- Reported to the host, in KiB, via HW_INFO2.TENSOR_MEM_KIB.
    g_tensor_bytes : positive;
    g_axi_addr_width : positive := 32
  );
  port (
    clk : in std_ulogic;
    reset : in std_ulogic := '0';

    --# {{}}
    s_axi_lite_m2s : in axi_lite_m2s_t;
    s_axi_lite_s2m : out axi_lite_s2m_t := axi_lite_s2m_init;

    --# {{}}
    -- Latched at the moment 'start' pulses (see the entity-level comment
    -- and 'program_base_addr_latch' below), not a live view of
    -- PROGRAM_BASE_ADDR's bus-side storage: the spec requires a write
    -- while BUSY to be accepted on the bus but to only take effect on the
    -- next START, i.e. a running program must not have its base address
    -- changed under it.
    program_base_addr : out std_ulogic_vector(g_axi_addr_width - 1 downto 0) := (others => '0');
    -- ISA v2.3 streaming-inference interface (spec section 6a): latched
    -- the same way as 'program_base_addr' above, except which source gets
    -- latched depends on WHICH start this is -- see 'input_output_addr_
    -- latch' below. A host write to INPUT_ADDR/OUTPUT_ADDR while BUSY='0'
    -- takes effect on the next START, exactly like PROGRAM_BASE_ADDR;
    -- while BUSY='1' it instead queues (or, if a job is already queued,
    -- is rejected with ERR_QUEUE_FULL -- see CTRL.START's own comment).
    input_addr : out std_ulogic_vector(g_axi_addr_width - 1 downto 0) := (others => '0');
    output_addr : out std_ulogic_vector(g_axi_addr_width - 1 downto 0) := (others => '0');
    -- Pulses exactly one cycle AFTER the accepted CTRL.START write, so that
    -- 'program_base_addr' above (registered, captured on that same write)
    -- is already stable when the consumer samples it. Handing 'start' out
    -- in the write cycle itself would make the consumer latch the previous
    -- base address -- 0 on the very first run, which fetches an all-zero
    -- descriptor at address 0 and decodes as an immediate HALT.
    start : out std_ulogic := '0';
    soft_reset_pulse : out std_ulogic := '0';

    --# {{}}
    seq_done : in std_ulogic;
    seq_error : in std_ulogic;
    err_code : in std_ulogic_vector(3 downto 0);
    err_pc : in std_ulogic_vector(31 downto 0);

    --# {{}}
    counters : in csr_counters_t;

    --# {{}}
    irq : out std_ulogic := '0'
  );
end entity cnn_accel_csr;

architecture a of cnn_accel_csr is

  signal regs_up : cnn_accel_regs_up_t := cnn_accel_regs_up_init;
  signal regs_down : cnn_accel_regs_down_t := cnn_accel_regs_down_init;
  signal reg_was_read : cnn_accel_reg_was_read_t := cnn_accel_reg_was_read_init;
  signal reg_was_written : cnn_accel_reg_was_written_t := cnn_accel_reg_was_written_init;

  -- STATUS-derived internal state (spec section 8: BUSY/DONE(sticky)/
  -- ERROR(sticky) + latched ERR_CODE/ERR_PC) plus the latched
  -- PROGRAM_BASE_ADDR. Reset by the external 'reset' only -- *not* by
  -- 'soft_reset_pulse' for DONE/ERROR/the latched fields, since an abort
  -- should not erase the diagnostic reason a prior run failed; BUSY does
  -- clear on 'soft_reset_pulse' since the abort really does stop whatever
  -- was running.
  signal busy_q : std_ulogic := '0';
  signal done_sticky_q : std_ulogic := '0';
  signal error_sticky_q : std_ulogic := '0';
  signal err_code_q : unsigned(3 downto 0) := (others => '0');
  signal err_pc_low_q : unsigned(15 downto 0) := (others => '0');
  signal program_base_addr_q : unsigned(g_axi_addr_width - 1 downto 0) := (others => '0');
  signal input_addr_q : unsigned(g_axi_addr_width - 1 downto 0) := (others => '0');
  signal output_addr_q : unsigned(g_axi_addr_width - 1 downto 0) := (others => '0');

  -- ISA v2.3 one-deep job queue (spec section 6a): CTRL.START written
  -- while BUSY='1' latches the bus-side INPUT_ADDR/OUTPUT_ADDR here
  -- instead of taking effect immediately, and 'auto_dispatch_i' below
  -- replays them the instant the running job's 'seq_done' fires.
  signal queued_q : std_ulogic := '0';
  signal queued_input_addr_q : unsigned(g_axi_addr_width - 1 downto 0) := (others => '0');
  signal queued_output_addr_q : unsigned(g_axi_addr_width - 1 downto 0) := (others => '0');

  signal start_i : std_ulogic;
  signal start_q : std_ulogic := '0';
  signal soft_reset_pulse_i : std_ulogic;

  -- CTRL.START written while already BUSY: latch it as the queued job
  -- ('queue_i') if the one-deep queue is empty, or reject it with
  -- ERR_QUEUE_FULL ('queue_full_error_i') if it is already full. Neither
  -- touches the job currently running.
  signal queue_i : std_ulogic;
  signal queue_full_error_i : std_ulogic;
  -- The running job's own completion, when a job was waiting: re-dispatch
  -- immediately, no second CTRL.START write needed.
  signal auto_dispatch_i : std_ulogic;

begin

  start <= start_q;
  soft_reset_pulse <= soft_reset_pulse_i;
  program_base_addr <= std_ulogic_vector(program_base_addr_q);
  input_addr <= std_ulogic_vector(input_addr_q);
  output_addr <= std_ulogic_vector(output_addr_q);

  -- CTRL.START/CTRL.ABORT: both are already exactly one-cycle-wide thanks
  -- to the register file's 'r_wpulse' storage (see the entity-level
  -- comment), so no extra pulse-shaping is needed here. A START write
  -- takes effect immediately while not BUSY; while BUSY, it queues
  -- ('queue_i') or is rejected ('queue_full_error_i') instead -- see the
  -- signal declarations above. ABORT is unconditional.
  start_i <= (regs_down.ctrl.start and not busy_q) or auto_dispatch_i;
  queue_i <= regs_down.ctrl.start and busy_q and not queued_q;
  queue_full_error_i <= regs_down.ctrl.start and busy_q and queued_q;
  auto_dispatch_i <= seq_done and queued_q;
  soft_reset_pulse_i <= regs_down.ctrl.abort;

  ------------------------------------------------------------------------
  -- BUSY / DONE / ERROR sticky-bit tracking, and ERR_CODE/ERR_PC_LOW
  -- latching (spec section 8).
  ------------------------------------------------------------------------

  status_tracking : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        busy_q <= '0';
        done_sticky_q <= '0';
        error_sticky_q <= '0';
        err_code_q <= (others => '0');
        err_pc_low_q <= (others => '0');
      else
        -- Order matters here (unlike before ISA v2.3): 'auto_dispatch_i'
        -- makes 'start_i' and 'seq_done' both '1' in the exact same
        -- cycle (the running job finishes and the queued one starts at
        -- once), and the LAST assignment below wins for that cycle. If
        -- the 'seq_done' clear ran last, BUSY would read '0' for the one
        -- cycle it takes 'start_q' to reach 'cnn_accel_cmd_proc', then
        -- never be set again for the job that is, in fact, now running.
        -- Putting 'start_i' last instead means a real completion with no
        -- queued job (the only case these two ever coincided in before
        -- v2.3) is unaffected, and a same-cycle auto-dispatch correctly
        -- leaves BUSY set. ABORT still wins over either: it stays last.
        if seq_done = '1' or seq_error = '1' then
          busy_q <= '0';
        end if;
        if start_i = '1' then
          busy_q <= '1';
        end if;
        if soft_reset_pulse_i = '1' then
          busy_q <= '0';
        end if;

        -- A same-cycle 'seq_done'/'seq_error' wins over a same-cycle
        -- write-1-to-clear, so the hardware set is the FIRST branch and the
        -- W1C is the 'elsif'. Getting this order wrong loses events: with
        -- the clear tested first, a run that finishes in the very cycle the
        -- host clears a stale DONE would leave DONE low forever, and the
        -- host would then wait on a completion that already happened.
        -- Set-wins costs nothing in the normal case (the two are almost
        -- never simultaneous) and turns an unresettable hang into at worst
        -- one redundant re-read of an already-observed status bit.
        if seq_done = '1' then
          done_sticky_q <= '1';
        elsif reg_was_written.status = '1' and regs_down.status.done = '1' then
          done_sticky_q <= '0';
        end if;

        -- Same set-wins ordering for ERROR. ERR_CODE/ERR_PC_LOW capture the
        -- FIRST error of a run only ('not error_sticky_q'): once an error is
        -- pending, a later 'seq_error' must not overwrite the diagnostic the
        -- host has not read yet, since the first failure is the one that
        -- explains the run and any later one may well be a consequence of it.
        -- 'queue_full_error_i' (ISA v2.3) joins the same set-wins path: it
        -- names a bad host write, not a bad program, so it carries no
        -- faulting PC (ERR_PC_LOW reads 0) and 'seq_error' wins if the two
        -- ever coincide -- a real program fault is the more useful of the
        -- two diagnoses to keep.
        if seq_error = '1' or queue_full_error_i = '1' then
          error_sticky_q <= '1';
          if error_sticky_q = '0' then
            if seq_error = '1' then
              err_code_q <= unsigned(err_code);
              err_pc_low_q <= unsigned(err_pc(15 downto 0));
            else
              err_code_q <= unsigned(c_err_queue_full);
              err_pc_low_q <= (others => '0');
            end if;
          end if;
        elsif reg_was_written.status = '1' and regs_down.status.error = '1' then
          error_sticky_q <= '0';
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- PROGRAM_BASE_ADDR latch: the bus-side register (regs_down, plain r_w)
  -- accepts writes at any time, but per the spec, a write while BUSY must
  -- only take effect on the next START -- so the value actually used by
  -- 'cmd_proc' is this entity's own latched copy, only updated when an
  -- accepted START pulses.
  ------------------------------------------------------------------------

  program_base_addr_latch : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        program_base_addr_q <= (others => '0');
      elsif start_i = '1' then
        program_base_addr_q <=
          unsigned(regs_down.program_base_addr.addr(g_axi_addr_width - 1 downto 0));
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- ISA v2.3 one-deep job queue (spec section 6a). Unlike PROGRAM_BASE_
  -- ADDR, INPUT_ADDR/OUTPUT_ADDR can be latched from two different
  -- sources at 'start_i': the live bus registers (a normal, not-BUSY
  -- START) or the queued pair (an auto-dispatch, 'auto_dispatch_i').
  -- 'queued_q' is cleared on 'soft_reset_pulse_i' too -- an ABORT drops
  -- the whole pipeline, not just the job currently running, so a queued
  -- job left behind can never auto-dispatch into a STATUS.QUEUED that
  -- would otherwise never clear.
  ------------------------------------------------------------------------

  queue_tracking : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        queued_q <= '0';
        queued_input_addr_q <= (others => '0');
        queued_output_addr_q <= (others => '0');
      else
        if queue_i = '1' then
          queued_q <= '1';
          queued_input_addr_q <=
            unsigned(regs_down.input_addr.addr(g_axi_addr_width - 1 downto 0));
          queued_output_addr_q <=
            unsigned(regs_down.output_addr.addr(g_axi_addr_width - 1 downto 0));
        elsif auto_dispatch_i = '1' then
          queued_q <= '0';
        end if;
        if soft_reset_pulse_i = '1' then
          queued_q <= '0';
        end if;
      end if;
    end if;
  end process;

  input_output_addr_latch : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        input_addr_q <= (others => '0');
        output_addr_q <= (others => '0');
      elsif start_i = '1' then
        if auto_dispatch_i = '1' then
          input_addr_q <= queued_input_addr_q;
          output_addr_q <= queued_output_addr_q;
        else
          input_addr_q <= unsigned(regs_down.input_addr.addr(g_axi_addr_width - 1 downto 0));
          output_addr_q <= unsigned(regs_down.output_addr.addr(g_axi_addr_width - 1 downto 0));
        end if;
      end if;
    end if;
  end process;

  -- The configuration snapshot above is a register, so it is only stable
  -- one cycle after the START write. 'start_q' delays the outgoing pulse by
  -- exactly that cycle; 'busy_q' is set in the same cycle as 'start_i', so
  -- the delay cannot let a second START through in between.
  start_delay : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        start_q <= '0';
      else
        start_q <= start_i;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Register read-side ('regs_up') values.
  ------------------------------------------------------------------------

  -- CTRL is write-only in spirit (self-clearing pulses); its 'regs_up'
  -- default ('cnn_accel_ctrl_init', all-zero) is never overridden, so it
  -- always reads back zero.

  regs_up.status.busy <= busy_q;
  regs_up.status.done <= done_sticky_q;
  regs_up.status.error <= error_sticky_q;
  regs_up.status.reserved0 <= (others => '0');
  regs_up.status.err_code <= err_code_q;
  regs_up.status.queued <= queued_q;
  regs_up.status.reserved1 <= (others => '0');
  regs_up.status.err_pc_low <= err_pc_low_q;

  -- HW_INFO/HW_INFO2: elaborated geometry, read from the generics actually
  -- instantiated into this bitstream rather than hardcoded, so one host
  -- driver binary works unmodified across build points (flow_status.md S3).
  regs_up.hw_info.pe_rows <= to_unsigned(g_pe_rows, cnn_accel_hw_info_pe_rows_width);
  regs_up.hw_info.pe_cols <= to_unsigned(g_pe_cols, cnn_accel_hw_info_pe_cols_width);
  regs_up.hw_info.tile_channels <=
    to_unsigned(g_tile_channels, cnn_accel_hw_info_tile_channels_width);
  regs_up.hw_info.max_kernel_size <=
    to_unsigned(g_max_kernel_size, cnn_accel_hw_info_max_kernel_size_width);

  regs_up.hw_info2.isa_version <=
    to_unsigned(cnn_accel_constant_isa_version, cnn_accel_hw_info2_isa_version_width);
  regs_up.hw_info2.tensor_mem_kib <=
    to_unsigned(g_tensor_bytes / 1024, cnn_accel_hw_info2_tensor_mem_kib_width);

  regs_up.hw_info3.max_pool_kernel_size <=
    to_unsigned(g_max_pool_kernel_size, cnn_accel_hw_info3_max_pool_kernel_size_width);
  regs_up.hw_info3.max_row_tile_words <=
    to_unsigned(g_max_row_tile_words, cnn_accel_hw_info3_max_row_tile_words_width);

  -- Counters: pure pass-through from the consolidated 'counters' input
  -- (csr_counters_t, src/cnn_accel_v2_pkg.vhd) into the individual
  -- read-only registers. 'cnn_accel_csr' samples these; it does not own
  -- any counter itself.
  regs_up.cmd_count.value <= unsigned(counters.cmd_count);
  regs_up.cycle_count.value <= unsigned(counters.cycle_count);
  regs_up.compute_cycles.value <= unsigned(counters.compute_cycles);
  regs_up.stall_cycles.value <= unsigned(counters.stall_cycles);
  regs_up.ddr_rd_bytes.value <= unsigned(counters.ddr_rd_bytes);
  regs_up.ddr_wr_bytes.value <= unsigned(counters.ddr_wr_bytes);
  regs_up.tensor_load_count.value <= unsigned(counters.tensor_load_count);
  regs_up.tensor_store_count.value <= unsigned(counters.tensor_store_count);
  regs_up.weight_load_bytes.value <= unsigned(counters.weight_load_bytes);
  regs_up.local_bytes.rd_kib <= unsigned(counters.local_rd_kib(15 downto 0));
  regs_up.local_bytes.wr_kib <= unsigned(counters.local_wr_kib(15 downto 0));

  ------------------------------------------------------------------------
  -- Register write-side ('regs_down') consumers.
  ------------------------------------------------------------------------

  irq <= (done_sticky_q and regs_down.irq_mask.done) or
    (error_sticky_q and regs_down.irq_mask.error);

  ------------------------------------------------------------------------
  -- The generated AXI4-Lite register file itself (regs_src/
  -- cnn_accel_register_file_axi_lite.vhd -- direct entity instantiation,
  -- per AGENTS.md; never a hand-declared register map).
  ------------------------------------------------------------------------

  axi_lite_register_file_inst : entity cnn_accel.cnn_accel_register_file_axi_lite
    port map (
      clk => clk,
      -- Unmodified external reset only, per the spec's reset policy: an
      -- abort ('soft_reset_pulse') must not erase 'PROGRAM_BASE_ADDR' or
      -- 'IRQ_MASK', and this reset clears every register's storage.
      reset => reset,
      axi_lite_m2s => s_axi_lite_m2s,
      axi_lite_s2m => s_axi_lite_s2m,
      regs_up => regs_up,
      regs_down => regs_down,
      reg_was_read => reg_was_read,
      reg_was_written => reg_was_written
    );

end architecture a;
