library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use std.textio.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;
use vunit_lib.python_pkg.all;
use vunit_lib.integer_array_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_isa_pkg.all;

-- Cross-language, bit-exact, end-to-end guard for cnn_accel_conv_core
-- ("M6b", see cnn_accel_conv_core.vhd's own header comment): drives the
-- REAL composed DUT (cnn_accel_window_gen -> cnn_accel_pe_array ->
-- cnn_accel_bias_requant, weight_buffer feeding pe_array) with the same
-- generated vector cases (generate_vectors.py from cnn_accel_model.py,
-- fetched live over VUnit's Python FFI by test/python_bridge/
-- conv_core_bridge.py -- see that module's own docstring and
-- shared/Vunit.md's "python_call and python_execute" section; no VHDL
-- file I/O happens in this testbench at all) that every other RTL-facing
-- consumer of that golden model uses, and compares its 'm_out' stream
-- against the expected bytes byte-for-byte -- no VHDL re-derivation of the
-- conv math anywhere in this file (unlike tb_cnn_accel_pe_array.vhd's own
-- independent 'golden_beat_partial'). This is deliberately the same
-- cross-language contract tb_cnn_accel_pe_array_from_vectors.vhd already
-- guards for cnn_accel_pe_array alone, but through the whole composed
-- datapath (activations in, requantized int8 activations out), so a defect
-- at any of the three internal handshake boundaries this composition adds
-- (window_gen->pe_array, pe_array->bias_requant, weight_buffer's read
-- ports) would show up here even if each submodule's own standalone
-- testbench stays green.
--
-- Scope (matches cnn_accel_conv_core.vhd's own excluded-scope list): only
-- 'CONV2D'/'FC' vector cases are run (the ones with packed weights --
-- 'DWCONV2D'/'POOL_MAX'/'POOL_AVG' need cnn_accel_pool or a different
-- weight layout not implemented by pack_weights_for_hw, see that
-- function's own docstring), and only cases whose 'out_channels <=
-- g_pe_rows' (every generated CONV2D/FC case already satisfies this --
-- layer-level output-channel tiling, re-streaming the whole ifmap once per
-- output-channel tile, is a future milestone cnn_accel_conv_core.vhd does
-- not implement, so this testbench cannot exercise more than one
-- output-channel tile either).
--
-- Tests: 'test_bitexact_backpressure'/'test_bitexact_full_throughput' both
-- run 'run_all_cases' -- the fixed, checked-in-name set of hand-authored
-- 'generate_vectors.py' cases (module_cnn_accel.py's per-'g_pe_rows'
-- config), selected by name (conv_core_bridge.py's 'select_hand_case').
-- 'test_bitexact_compiler_cases' (M10, doc/tosa_compiler_plan.md ~line
-- 613) instead runs 'run_compiler_cases', which reads 'vectors_root &
-- "/cases.txt"' (the one file this testbench still reads directly -- a
-- plain manifest of case names, not vector data) and selects each case by
-- directory (conv_core_bridge.py's 'select_dir_case') -- that root is
-- populated by module_cnn_accel.py's compiler-vectors pre_config hook,
-- which compiles real TOSA fixtures with 'cnnc' and writes their per-layer
-- vectors via 'cnnc.backend.cnn_accel_v1.vectors.write_conv_core_vectors'
-- (the compiler's OWN emitted weights/program output, not
-- 'cnn_accel_model' called directly, is what is bit-exact-checked here),
-- one config only at the default 'g_pe_rows'. A missing or empty
-- 'cases.txt' is a hard 'severity failure', not a vacuous pass -- see
-- 'run_compiler_cases' below.
--
-- Weight/bias preload: this testbench IS the "future cnn_accel_axi_read_
-- dma instance" cnn_accel_conv_core.vhd's own header comment anticipates
-- for driving 'fill_start'/'fill_is_bias'/'s_weight' -- the packed weights
-- are streamed one int8 lane per fill beat (cnn_accel_weight_buffer.vhd's
-- own contract: one lane per accepted beat, auto-advancing), in the exact
-- flat order cnn_accel_model.pack_weights_for_hw already produced it (that
-- function's docstring: this is precisely the order 'weight_rd_addr'
-- walks), so no reordering happens on this side either. The bias (LOGICAL,
-- unpadded, 'out_channels' int32 values) is zero-padded up to 'g_pe_rows'
-- lanes here before streaming (D11 -- no packed-bias vector exists for
-- these cases, so this is the one place this testbench does its own
-- trivial, shape-only, zero-padding, not conv math).
-- ISA v1.2 (H2): a case with FLAG_PER_CHANNEL_EN additionally streams its
-- scale table (already padded to 'g_pe_rows' entries, two records per
-- entry) one entry per 'fill_is_scale' beat right after the bias -- the
-- same tile-load phase, into the weight_buffer's scale region.
--
-- Fill sessions: every case pulses 'fill_start' once before streaming its
-- own weight/bias set (see run_selected_case), which is what makes
-- cnn_accel_weight_buffer's "new fill session" (write pointers reset to
-- 0) actually fire between cases -- single-buffered now, so there is no
-- bank to alternate any more; each case's fill fully overwrites the
-- previous case's weight/bias set before that case's own 'start' pulse.
entity tb_cnn_accel_conv_core is
  generic (
    -- Independent per-link randomized-backpressure generics, swept per
    -- test in module_cnn_accel.py's setup_vunit (0/0 for the dedicated
    -- full-throughput test, nonzero otherwise) -- mirrors every other
    -- cnn_accel testbench's identical in/out generic pair. 's_stream' is
    -- the one input link stalled; 'm_out' the one output link.
    stall_probability_percent_in : natural := 20;
    stall_probability_percent_out : natural := 20;
    -- VUnit's own per-test-config output directory, filled in by VUnit
    -- itself. Only 'test_bitexact_compiler_cases' still uses it: module_
    -- cnn_accel.py's '_compiler_vectors_pre_config' hook writes that
    -- config's compiler-generated vectors here before simulation starts
    -- ('run_compiler_cases' reads 'cases.txt' from it and selects each
    -- case by directory). The hand-authored cases 'run_all_cases' selects
    -- by name need no directory at all -- conv_core_bridge.py builds them
    -- into its own private scratch directory, at THIS config's
    -- 'g_pe_rows' (run_selected_case checks every selected case's
    -- 'pe_rows' field against it either way).
    output_path : string;
    -- THE single scaling knob (flow_status.md S1-S7). Default is the
    -- shipped 8; module_cnn_accel.py adds a second config at the
    -- CI-proven 16 (cnn_accel_constants.PE_ROWS_SCALED) with a matching
    -- 'vectors_root'. Not read from the generated cnn_accel_regs_pkg here
    -- on purpose -- this testbench must be able to run at the non-default
    -- legal value, which is exactly what the generated constant is not.
    g_pe_rows : positive := 8;
    runner_cfg : string
  );
end entity tb_cnn_accel_conv_core;

architecture tb of tb_cnn_accel_conv_core is

  ------------------------------------------------------------------------
  -- DUT generics -- the reference hardware configuration point
  -- (doc/cnn_accel_tiled_dataflow_proposal.md section 7 / module_cnn_
  -- accel.py's own '_PE_ROWS'/'_PE_COLS'/'_TILE_CHANNELS'), fixed for
  -- this whole testbench except 'c_pe_rows', which is the 'g_pe_rows'
  -- scaling knob (only runtime 'cfg_*' values vary between cases,
  -- exactly like a real compiled program would). 'c_max_row_tile_
  -- words'/'c_weight_buffer_depth' are sized generously for the small
  -- vector-case shapes actually exercised here (not the full '_MAX_ROW_
  -- TILE_WORDS'/'_WEIGHT_BUFFER_DEPTH' netlist-build values), for fast
  -- simulation.
  ------------------------------------------------------------------------

  constant c_pe_rows : positive := g_pe_rows;
  -- The generated vector root for this config (see 'output_path').
  constant vectors_root : string := output_path;
  constant c_pe_cols : positive := 8;
  constant c_tile_channels : positive := 8;
  constant c_max_kernel_size : positive := 3;
  constant c_accum_width : positive := 32;
  constant c_max_row_tile_words : positive := 64;
  constant c_weight_buffer_depth : positive := 64;
  -- Independent of 'c_weight_buffer_depth' -- every generated case has
  -- 'out_channels <= g_pe_rows' (run_case's own assert), so only bias row
  -- 0 is ever read; a handful of rows is generous headroom.
  constant c_bias_buffer_depth : positive := 4;

  constant c_weight_lanes : positive := c_pe_rows * c_pe_cols;

  constant c_opcode_conv2d : natural := to_integer(unsigned(OPCODE_CONV2D));
  constant c_opcode_fc : natural := to_integer(unsigned(OPCODE_FC));

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal cfg_kernel_h : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_kernel_w : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_stride_h : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_stride_w : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pad_top : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pad_bottom : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pad_left : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_pad_right : std_ulogic_vector(7 downto 0) := (others => '0');
  -- ISA v2.1: the signed int8 value padded taps take (the input tensor's
  -- quantization zero-point), read from each case's desc.txt.
  signal cfg_pad_value : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_in_width : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_in_height : std_ulogic_vector(15 downto 0) := (others => '0');
  -- Pre-computed output frame dimensions (see the DUT's port comment):
  -- driven from this testbench's own 'v_out_width'/'v_out_height', the
  -- same values the expected-output model is built from.
  signal cfg_out_width : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_out_height : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_in_channels : std_ulogic_vector(15 downto 0) := (others => '0');

  signal cfg_bias_en : std_ulogic := '0';
  signal cfg_requant_en : std_ulogic := '0';
  signal cfg_relu_en : std_ulogic := '0';
  signal cfg_requant_scale : std_ulogic_vector(31 downto 0) := (others => '0');
  signal cfg_requant_shift : std_ulogic_vector(7 downto 0) := (others => '0');
  -- ISA v1.1 (H1) epilogue fields, from desc.txt's output_offset/
  -- clamp_min/clamp_max records and FLAG_CLAMP_EN (flags bit 4).
  signal cfg_output_offset : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_clamp_en : std_ulogic := '0';
  signal cfg_clamp_min : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_clamp_max : std_ulogic_vector(7 downto 0) := (others => '0');
  -- ISA v1.2 (H2) FLAG_PER_CHANNEL_EN (flags bit 5): the per-lane
  -- (multiplier, shift) table is streamed from 'scale_table_packed.txt'
  -- into cnn_accel_weight_buffer's scale region with 'fill_is_scale'.
  signal cfg_per_channel_en : std_ulogic := '0';

  signal start : std_ulogic := '0';
  signal done : std_ulogic;

  signal s_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_stream_s2m : axi_stream_s2m_t;

  signal fill_start : std_ulogic := '0';
  signal fill_is_bias : std_ulogic := '0';
  signal fill_is_scale : std_ulogic := '0';
  signal s_weight_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_weight_s2m : axi_stream_s2m_t;

  signal m_out_m2s : axi_stream_m2s_t;
  signal m_out_s2m : axi_stream_s2m_t := (ready => '0');

  -- Non-blocking scoreboard for the single 'm_out' output link: one
  -- (packed data, last) pair pushed per expected pixel, popped and
  -- checked by 'monitor_out' on every accepted 'm_out' beat -- mirrors
  -- tb_cnn_accel_pe_array.vhd's identical 'expected_q' idiom.
  constant expected_q : queue_t := new_queue;

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  ------------------------------------------------------------------------
  -- 'desc_fields' index constants: fixed positions in the flat array
  -- test/python_bridge/conv_core_bridge.py's 'get_desc_fields()' returns
  -- (that module's own '_DESC_FIELD_ORDER', plus 'tile_channels'/
  -- 'pe_rows' appended after it) -- named indices instead of magic
  -- numbers at each 'get(desc_fields, N)' call site below.
  ------------------------------------------------------------------------

  constant c_df_opcode : natural := 0;
  constant c_df_flags : natural := 1;
  constant c_df_in_width : natural := 2;
  constant c_df_in_height : natural := 3;
  constant c_df_in_channels : natural := 4;
  constant c_df_out_channels : natural := 5;
  constant c_df_kernel_h : natural := 6;
  constant c_df_kernel_w : natural := 7;
  constant c_df_stride_h : natural := 8;
  constant c_df_stride_w : natural := 9;
  constant c_df_pad_top : natural := 10;
  constant c_df_pad_bottom : natural := 11;
  constant c_df_pad_left : natural := 12;
  constant c_df_pad_right : natural := 13;
  constant c_df_pad_value : natural := 14;
  constant c_df_requant_scale : natural := 15;
  constant c_df_requant_shift : natural := 16;
  constant c_df_output_offset : natural := 17;
  constant c_df_clamp_min : natural := 18;
  constant c_df_clamp_max : natural := 19;
  constant c_df_tile_channels : natural := 20;
  constant c_df_pe_rows : natural := 21;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_conv_core
    generic map (
      g_pe_rows => c_pe_rows,
      g_pe_cols => c_pe_cols,
      g_accum_width => c_accum_width,
      g_max_kernel_size => c_max_kernel_size,
      g_tile_channels => c_tile_channels,
      g_max_row_tile_words => c_max_row_tile_words,
      g_weight_buffer_depth => c_weight_buffer_depth,
      g_bias_buffer_depth => c_bias_buffer_depth
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_kernel_h => cfg_kernel_h,
      cfg_kernel_w => cfg_kernel_w,
      cfg_stride_h => cfg_stride_h,
      cfg_stride_w => cfg_stride_w,
      cfg_pad_top => cfg_pad_top,
      cfg_pad_bottom => cfg_pad_bottom,
      cfg_pad_left => cfg_pad_left,
      cfg_pad_right => cfg_pad_right,
      cfg_pad_value => cfg_pad_value,
      cfg_in_width => cfg_in_width,
      cfg_in_height => cfg_in_height,
      cfg_out_width => cfg_out_width,
      cfg_out_height => cfg_out_height,
      cfg_in_channels => cfg_in_channels,

      cfg_bias_en => cfg_bias_en,
      cfg_requant_en => cfg_requant_en,
      cfg_relu_en => cfg_relu_en,
      cfg_requant_scale => cfg_requant_scale,
      cfg_requant_shift => cfg_requant_shift,
      cfg_output_offset => cfg_output_offset,
      cfg_clamp_en => cfg_clamp_en,
      cfg_clamp_min => cfg_clamp_min,
      cfg_clamp_max => cfg_clamp_max,
      cfg_per_channel_en => cfg_per_channel_en,

      start => start,
      done => done,

      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,

      fill_start => fill_start,
      fill_is_bias => fill_is_bias,
      fill_is_scale => fill_is_scale,
      s_weight_m2s => s_weight_m2s,
      s_weight_s2m => s_weight_s2m,

      m_out_m2s => m_out_m2s,
      m_out_s2m => m_out_s2m
    );

  ------------------------------------------------------------------------
  -- Output monitor: independent randomized-'ready' responder. Pops and
  -- checks the scoreboard queue on every accepted 'm_out' beat -- mirrors
  -- tb_cnn_accel_pe_array.vhd's identical 'monitor_out' process.
  ------------------------------------------------------------------------

  monitor_out : process
    variable rnd : RandomPType;
    variable expected_bytes : std_ulogic_vector(8 * c_pe_rows - 1 downto 0);
    variable expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_monitor");
    m_out_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < stall_probability_percent_out then
        m_out_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 4) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_out_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_out_m2s.valid = '1' and m_out_s2m.ready = '1' then
        expected_bytes := pop(expected_q);
        expected_last := pop(expected_q);
        check_equal(
          m_out_m2s.data(8 * c_pe_rows - 1 downto 0), expected_bytes,
          "m_out data mismatch (Python golden model vs. real composed RTL)"
        );
        check_equal(m_out_m2s.last, expected_last, "m_out last mismatch");
      end if;
    end loop;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;
    variable discard : integer;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
    end procedure;

    -- Waits (bounded) until 'expected_q' has drained, then confirms it is
    -- truly empty (every expected output pixel actually arrived) --
    -- mirrors tb_cnn_accel_pe_array.vhd's identical 'drain_and_check'.
    procedure drain_and_check(max_wait_cycles : positive) is
      variable cycles : natural := 0;
    begin
      while not is_empty(expected_q) and cycles < max_wait_cycles loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
      end loop;
      check_true(is_empty(expected_q), "m_out scoreboard queue did not drain in time");
    end procedure;

    -- Runs the currently-selected case (test/python_bridge/
    -- conv_core_bridge.py's 'select_hand_case'/'select_dir_case' --
    -- see 'run_hand_case'/'run_dir_case' below) end to end: preload
    -- weight_buffer (weights then bias) after a fresh 'fill_start' pulse,
    -- configure and start the DUT, push every pixel's expected output,
    -- stream every input beat with randomized backpressure, then drain
    -- and check. See this file's own header comment for the full
    -- rationale of each step. 'case_label' names the case in every
    -- assertion message only -- selection already happened before this
    -- is called.
    procedure run_selected_case(case_label : string) is
      variable desc_fields : integer_array_t := python_call("get_desc_fields");
      variable opcode : integer := get(desc_fields, c_df_opcode);
      variable flags : integer := get(desc_fields, c_df_flags);
      variable v_in_width : integer := get(desc_fields, c_df_in_width);
      variable v_in_height : integer := get(desc_fields, c_df_in_height);
      variable v_in_channels : integer := get(desc_fields, c_df_in_channels);
      variable v_out_channels : integer := get(desc_fields, c_df_out_channels);
      variable v_kernel_h : integer := get(desc_fields, c_df_kernel_h);
      variable v_kernel_w : integer := get(desc_fields, c_df_kernel_w);
      variable v_stride_h : integer := get(desc_fields, c_df_stride_h);
      variable v_stride_w : integer := get(desc_fields, c_df_stride_w);
      variable v_pad_top : integer := get(desc_fields, c_df_pad_top);
      variable v_pad_bottom : integer := get(desc_fields, c_df_pad_bottom);
      variable v_pad_left : integer := get(desc_fields, c_df_pad_left);
      variable v_pad_right : integer := get(desc_fields, c_df_pad_right);
      variable v_pad_value : integer := get(desc_fields, c_df_pad_value);
      variable v_requant_scale : integer := get(desc_fields, c_df_requant_scale);
      variable v_requant_shift : integer := get(desc_fields, c_df_requant_shift);
      variable v_output_offset : integer := get(desc_fields, c_df_output_offset);
      variable v_clamp_min : integer := get(desc_fields, c_df_clamp_min);
      variable v_clamp_max : integer := get(desc_fields, c_df_clamp_max);
      variable v_pe_rows : integer := get(desc_fields, c_df_pe_rows);
      variable v_tile_channels : integer := get(desc_fields, c_df_tile_channels);

      variable v_n_tiles : positive := (v_in_channels + c_tile_channels - 1) / c_tile_channels;
      variable v_out_width : positive :=
        (v_in_width + v_pad_left + v_pad_right - v_kernel_w) / v_stride_w + 1;
      variable v_out_height : positive :=
        (v_in_height + v_pad_top + v_pad_bottom - v_kernel_h) / v_stride_h + 1;
      variable v_num_pixels : positive := v_out_width * v_out_height;
      variable v_n_weight_vals : positive :=
        v_n_tiles * v_kernel_h * v_kernel_w * c_weight_lanes;

      -- flags bit 5 = FLAG_PER_CHANNEL_EN (ISA v1.2, H2).
      variable v_per_channel_en : boolean := (flags / 32) mod 2 = 1;

      variable weights_flat : integer_array_t;
      variable bias_flat : integer_array_t;
      -- 'scale_table_packed' (doc/cnn_accel_test_vectors.md): two records
      -- (multiplier, shift) per PADDED entry, 'c_pe_rows' entries for the
      -- single output-channel tile this procedure supports. Only fetched
      -- when 'v_per_channel_en'.
      variable scale_flat : integer_array_t;
      variable input_flat : integer_array_t;
      variable expected_flat : integer_array_t;

      variable tile_data : std_ulogic_vector(8 * c_tile_channels - 1 downto 0);
      variable expected_bytes : std_ulogic_vector(8 * c_pe_rows - 1 downto 0);
      variable ic : natural;
    begin
      assert opcode = c_opcode_conv2d or opcode = c_opcode_fc
        report "tb_cnn_accel_conv_core: run_selected_case only supports CONV2D/FC vector " &
          "cases (pack_weights_for_hw's own ratified scope), got opcode=" &
          integer'image(opcode) & " from '" & case_label & "'"
        severity failure;

      assert v_out_channels <= c_pe_rows
        report "tb_cnn_accel_conv_core: run_selected_case only supports a single " &
          "output-channel tile (out_channels <= g_pe_rows) -- layer-level output-channel " &
          "tiling is a future milestone, see cnn_accel_conv_core.vhd's own entity-level " &
          "comment; got out_channels=" & integer'image(v_out_channels) & " from '" &
          case_label & "'"
        severity failure;

      -- 'weights_packed' is only meaningful for the packing point it was
      -- generated at (D10: rows of 'pe_rows*tile_channels' lanes) -- a
      -- mismatch here would not be a datapath defect but a wrong
      -- 'g_pe_rows'/case pairing, so fail loudly and by name.
      assert v_pe_rows = c_pe_rows
        report "tb_cnn_accel_conv_core: '" & case_label & "' was packed at pe_rows=" &
          integer'image(v_pe_rows) & " but this testbench runs at g_pe_rows=" &
          integer'image(c_pe_rows) & " -- wrong case/config pairing"
        severity failure;
      assert v_tile_channels = c_tile_channels
        report "tb_cnn_accel_conv_core: '" & case_label & "' was packed at tile_channels=" &
          integer'image(v_tile_channels) & " but this testbench runs at tile_channels=" &
          integer'image(c_tile_channels)
        severity failure;

      weights_flat := python_call("get_weights_packed_flat");
      bias_flat := python_call("get_bias_flat");
      input_flat := python_call("get_input_flat");
      expected_flat := python_call("get_expected_flat");
      if v_per_channel_en then
        scale_flat := python_call("get_scale_table_packed_flat");
      end if;

      -- Start a new fill session (write pointers reset to 0) before
      -- streaming any fill beat -- see this file's header comment. One
      -- idle cycle after the pulse keeps it unambiguous with respect to
      -- the first fill beat below (a beat presented the same cycle as
      -- 'fill_start' is not accepted).
      fill_start <= '1';
      wait until rising_edge(clk);
      fill_start <= '0';
      wait until rising_edge(clk);

      -- Fill weights: one int8 lane per beat, 'fill_is_bias'='0', in
      -- cnn_accel_model.pack_weights_for_hw's own flat order (that
      -- function's docstring: exactly the order 'weight_rd_addr' walks).
      fill_is_bias <= '0';
      for i in 0 to v_n_weight_vals - 1 loop
        s_weight_m2s.data(7 downto 0) <= std_ulogic_vector(to_signed(get(weights_flat, i), 8));
        s_weight_m2s.data(axi_stream_data_sz - 1 downto 8) <= (others => '0');
        s_weight_m2s.valid <= '1';
        wait until rising_edge(clk) and s_weight_s2m.ready = '1';
      end loop;
      s_weight_m2s.valid <= '0';

      -- Fill bias: one int32 lane per beat, 'fill_is_bias'='1', zero-
      -- padded up to 'c_pe_rows' lanes for output channels beyond
      -- 'v_out_channels' (D11 -- see this file's header comment; no
      -- packed-bias vector file exists for these cases).
      fill_is_bias <= '1';
      for i in 0 to c_pe_rows - 1 loop
        if i < v_out_channels then
          s_weight_m2s.data(31 downto 0) <= std_ulogic_vector(to_signed(get(bias_flat, i), 32));
        else
          s_weight_m2s.data(31 downto 0) <= (others => '0');
        end if;
        s_weight_m2s.data(axi_stream_data_sz - 1 downto 32) <= (others => '0');
        s_weight_m2s.valid <= '1';
        wait until rising_edge(clk) and s_weight_s2m.ready = '1';
      end loop;
      s_weight_m2s.valid <= '0';
      fill_is_bias <= '0';

      -- Fill the per-channel scale table (ISA v1.2, H2) in the same
      -- tile-load phase, one 'c_scale_entry_width'-bit entry per beat with
      -- 'fill_is_scale'='1': multiplier in bits [31:0], shift in [39:32],
      -- exactly cnn_accel_model.pack_scale_table_for_hw's 8-byte entry
      -- with the three reserved bytes dropped. The file is already padded
      -- to 'c_pe_rows' entries, so no zero-padding happens here.
      if v_per_channel_en then
        fill_is_scale <= '1';
        for i in 0 to c_pe_rows - 1 loop
          s_weight_m2s.data(c_scale_entry_mult_width - 1 downto 0) <=
            std_ulogic_vector(to_signed(get(scale_flat, 2 * i), c_scale_entry_mult_width));
          s_weight_m2s.data(c_scale_entry_width - 1 downto c_scale_entry_mult_width) <=
            std_ulogic_vector(to_unsigned(get(scale_flat, 2 * i + 1), c_scale_entry_shift_width));
          s_weight_m2s.data(axi_stream_data_sz - 1 downto c_scale_entry_width) <= (others => '0');
          s_weight_m2s.valid <= '1';
          wait until rising_edge(clk) and s_weight_s2m.ready = '1';
        end loop;
        s_weight_m2s.valid <= '0';
        fill_is_scale <= '0';
      end if;

      -- Configure and pulse 'start'.
      cfg_kernel_h <= std_ulogic_vector(to_unsigned(v_kernel_h, 8));
      cfg_kernel_w <= std_ulogic_vector(to_unsigned(v_kernel_w, 8));
      cfg_stride_h <= std_ulogic_vector(to_unsigned(v_stride_h, 8));
      cfg_stride_w <= std_ulogic_vector(to_unsigned(v_stride_w, 8));
      cfg_pad_top <= std_ulogic_vector(to_unsigned(v_pad_top, 8));
      cfg_pad_bottom <= std_ulogic_vector(to_unsigned(v_pad_bottom, 8));
      cfg_pad_left <= std_ulogic_vector(to_unsigned(v_pad_left, 8));
      cfg_pad_right <= std_ulogic_vector(to_unsigned(v_pad_right, 8));
      -- Signed, unlike the four pad counts above.
      cfg_pad_value <= std_ulogic_vector(to_signed(v_pad_value, 8));
      cfg_in_width <= std_ulogic_vector(to_unsigned(v_in_width, 16));
      cfg_in_height <= std_ulogic_vector(to_unsigned(v_in_height, 16));
      cfg_in_channels <= std_ulogic_vector(to_unsigned(v_in_channels, 16));
      cfg_out_width <= std_ulogic_vector(to_unsigned(v_out_width, 16));
      cfg_out_height <= std_ulogic_vector(to_unsigned(v_out_height, 16));
      cfg_bias_en <= to_sl((flags / 2) mod 2 = 1);
      cfg_requant_en <= to_sl((flags / 4) mod 2 = 1);
      cfg_relu_en <= to_sl(flags mod 2 = 1);
      cfg_requant_scale <= std_ulogic_vector(to_signed(v_requant_scale, 32));
      cfg_requant_shift <= std_ulogic_vector(to_unsigned(v_requant_shift, 8));
      cfg_clamp_en <= to_sl((flags / 16) mod 2 = 1);
      cfg_output_offset <= std_ulogic_vector(to_signed(v_output_offset, 16));
      cfg_clamp_min <= std_ulogic_vector(to_signed(v_clamp_min, 8));
      cfg_clamp_max <= std_ulogic_vector(to_signed(v_clamp_max, 8));
      cfg_per_channel_en <= to_sl(v_per_channel_en);

      wait until rising_edge(clk);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';

      -- Push every pixel's expected output up front (padded to
      -- 'c_pe_rows' bytes with 0 beyond 'v_out_channels' -- exact, not
      -- approximate, since a padded PE row's weight/bias are both 0, see
      -- this file's header comment).
      for p in 0 to v_num_pixels - 1 loop
        expected_bytes := (others => '0');
        for oc in 0 to v_out_channels - 1 loop
          expected_bytes(8 * (oc + 1) - 1 downto 8 * oc) :=
            std_ulogic_vector(to_signed(get(expected_flat, p * v_out_channels + oc), 8));
        end loop;
        push(expected_q, expected_bytes);
        push(expected_q, to_sl(p = v_num_pixels - 1));
      end loop;

      -- Feed 's_stream': raster order (row, col), 'v_n_tiles' beats per
      -- pixel, randomized input-side stall -- mirrors tb_cnn_accel_
      -- pe_array.vhd's/tb_cnn_accel_window_gen.vhd's identical
      -- 'send_*_beat' idiom.
      for row in 0 to v_in_height - 1 loop
        for col in 0 to v_in_width - 1 loop
          for tile in 0 to v_n_tiles - 1 loop
            tile_data := (others => '0');
            for c in 0 to c_tile_channels - 1 loop
              ic := tile * c_tile_channels + c;
              if ic < v_in_channels then
                tile_data(8 * (c + 1) - 1 downto 8 * c) :=
                  std_ulogic_vector(to_signed(
                    get(input_flat, (row * v_in_width + col) * v_in_channels + ic), 8
                  ));
              end if;
            end loop;

            if rnd.RandInt(0, 99) < stall_probability_percent_in then
              s_stream_m2s.valid <= '0';
              for i in 1 to rnd.RandInt(1, 4) loop
                wait until rising_edge(clk);
              end loop;
            end if;

            s_stream_m2s.data(8 * c_tile_channels - 1 downto 0) <= tile_data;
            s_stream_m2s.data(axi_stream_data_sz - 1 downto 8 * c_tile_channels) <=
              (others => '0');
            s_stream_m2s.valid <= '1';
            wait until rising_edge(clk) and s_stream_s2m.ready = '1';
          end loop;
        end loop;
      end loop;
      s_stream_m2s.valid <= '0';

      drain_and_check(20000);
    end procedure;

    -- Selects one of generate_vectors.py's hand-authored cases by name
    -- (test/python_bridge/conv_core_bridge.py's own case registry, built
    -- once per simulation process at this config's 'g_pe_rows') and runs
    -- it -- for 'run_all_cases' below.
    procedure run_hand_case(name : string) is
    begin
      discard := python_call(
        "select_hand_case", arg => string'(name), kwargs => kw("pe_rows", c_pe_rows)
      );
      run_selected_case(name);
    end procedure;

    -- Selects a case already written to 'case_dir' on disk (module_cnn_
    -- accel.py's '_compiler_vectors_pre_config', unchanged) and runs it --
    -- for 'run_compiler_cases' below.
    procedure run_dir_case(case_dir : string) is
    begin
      discard := python_call("select_dir_case", arg => string'(case_dir));
      run_selected_case(case_dir);
    end procedure;

    -- Reads 'vectors_root & "/cases.txt"' (one case name per line) and
    -- runs every case listed, in file order -- the compiler-generated
    -- counterpart of 'run_all_cases' below, for 'test_bitexact_compiler_
    -- cases' (see this file's own header comment). Fails loudly (rather
    -- than passing vacuously) if 'cases.txt' is missing, or is present
    -- but lists zero cases -- a config whose pre_config hook did not
    -- actually write any vectors must not report a green test.
    procedure run_compiler_cases is
      file f : text;
      variable l : line;
      variable status : file_open_status;
      variable n_cases : natural := 0;
    begin
      file_open(status, f, vectors_root & "/cases.txt", read_mode);
      assert status = open_ok
        report "tb_cnn_accel_conv_core: could not open '" & vectors_root &
          "/cases.txt' -- module_cnn_accel.py's compiler-vectors pre_config hook " &
          "did not write it"
        severity failure;

      while not endfile(f) loop
        readline(f, l);
        if l'length > 0 then
          n_cases := n_cases + 1;
          run_dir_case(vectors_root & "/" & l.all);
        end if;
      end loop;
      file_close(f);

      assert n_cases > 0
        report "tb_cnn_accel_conv_core: '" & vectors_root &
          "/cases.txt' listed zero cases -- a config that runs no cases must not pass silently"
        severity failure;
    end procedure;

    -- Runs every generated CONV2D/FC vector case (doc/cnn_accel_test_vectors.md's
    -- table), each with its own fresh 'fill_start' session -- see this
    -- file's header comment. Called identically by both tests below;
    -- only the stall generics (module_cnn_accel.py's per-test config)
    -- differ between them.
    procedure run_all_cases is
    begin
      run_hand_case("conv1x1_c4_o4");
      run_hand_case("conv3x3_s1_c3_o8_pad1");
      run_hand_case("conv3x3_s2_c8_o8_pad1");
      run_hand_case("conv3x3_s1_extremes");
      run_hand_case("conv3x3_asymmetric_pad");
      run_hand_case("conv3x3_negative_requant_scale");
      run_hand_case("fc_in6_out4");
      run_hand_case("conv3x3_c20_o6_multitile");
      -- ISA v2.1 'pad_value' on the CONV path: a padded tap takes the
      -- input tensor's quantization zero-point, not 0. Both cases use
      -- all-negative input against positive weights so the difference
      -- survives requantization instead of being clipped away -- see
      -- generate_vectors.py's own comment, and note that a case which
      -- does NOT do that passes against a 'cfg_pad_value' tied to zero.
      run_hand_case("conv3x3_pad_zero_point");
      run_hand_case("conv3x3_pad_value_asymmetric");
      -- The one case that drives EVERY lane of a 16-row array (out_channels
      -- = 16, the target backbone's layer 1). It only exists in the scaled
      -- vector root -- at g_pe_rows=8 it would be two output-channel
      -- tiles, which this composition cannot be driven with standalone
      -- (run_selected_case's own out_channels <= g_pe_rows assert), so it
      -- is gated on the generic rather than on the case set's contents.
      if c_pe_rows >= 16 then
        run_hand_case("conv3x3_c8_o16");
      end if;
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));
    python_execute(file_name => tb_path(runner_cfg) & "python_bridge/conv_core_bridge.py");

    do_reset;
    wait until rising_edge(clk);

    if run("test_bitexact_backpressure") then
      -- Randomized independent per-link backpressure (generic-driven,
      -- nonzero by module_cnn_accel.py's per-test config) across every
      -- checked-in CONV2D/FC case, including the multi-tile case (T=3)
      -- that exercises the window_gen->pe_array first_tile/last_tile
      -- partial-sum-carry path end to end.
      run_all_cases;

    elsif run("test_bitexact_full_throughput") then
      -- Same cases, zero stall on both links (module_cnn_accel.py sets
      -- both stall generics to 0 for this test only) -- proves the
      -- composition sustains back-to-back beats with no artificially
      -- introduced bubble, on top of (not instead of) the correctness
      -- check every case already performs.
      run_all_cases;

    elsif run("test_bitexact_compiler_cases") then
      -- Compiler-generated cases (M10, doc/tosa_compiler_plan.md ~line
      -- 613): module_cnn_accel.py's compiler-vectors pre_config hook
      -- compiles real TOSA fixtures with cnnc and writes their per-layer
      -- vectors via cnnc.backend.cnn_accel_v1.vectors.
      -- write_conv_core_vectors straight into this config's own
      -- 'output_path' (one combined 'cases.txt', case dirs distinguished
      -- by a per-fixture name prefix), one config only, at the default
      -- 'g_pe_rows' (the compiler packs weights at the target's own
      -- discovered internal_tiling, PE_ROWS -- see that hook's own
      -- comment). Nonzero backpressure on both links, same as
      -- 'test_bitexact_backpressure'.
      run_compiler_cases;

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
