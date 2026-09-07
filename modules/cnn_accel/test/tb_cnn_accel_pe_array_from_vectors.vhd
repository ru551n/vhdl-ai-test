library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use std.textio.all;

library vunit_lib;
context vunit_lib.vunit_context;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Cross-language bit-exactness guard for the D10 weight-lane-order defect
-- (see cnn_accel_model.pack_weights_for_hw's docstring and
-- cnn_accel_pe_array.vhd's compute_partial_sums() comment): unlike
-- tb_cnn_accel_pe_array.vhd, whose golden model (golden_beat_partial) is an
-- independent VHDL re-derivation that agrees with this DUT by
-- construction, THIS testbench drives the real cnn_accel_pe_array with a
-- packed weight image and expected accumulator results that both come
-- straight out of cnn_accel_model.py (via generate_vectors.py's
-- pe_array_xlang_check case) -- exercising the actual Python-packer ->
-- RTL-reader contract end to end, not two independently-written models
-- that happen to agree.
--
-- Kept deliberately minimal: a single pointwise (1x1 kernel) CONV2D shape
-- (module_cnn_accel.py's own reference hardware point, g_pe_rows=
-- g_pe_cols=g_tile_channels=8), one output-channel tile only (out_channels
-- <= g_pe_rows -- the only shape this DUT can be exercised with
-- standalone; multiple output-channel tiles need the whole ifmap
-- re-streamed by the not-yet-built layer_ctrl/DMA, D6 section 5), with a
-- partial input-channel tile (in_channels=10, not a multiple of
-- g_tile_channels=8) and a partial output-channel tile (out_channels=5 <
-- g_pe_rows=8) both exercised -- D11's zero-padding cases. No
-- backpressure sweep here (tb_cnn_accel_pe_array.vhd already covers that
-- against its own golden model); this testbench's only job is the
-- cross-language data contract.
--
-- The shape constants below (c_pe_rows, c_pe_cols, c_tile_channels,
-- c_in_w/h/c, c_out_c) are NOT read from desc.txt -- they are fixed to
-- exactly match generate_vectors.py's pe_array_xlang_check case (and
-- module_cnn_accel.py's HW_TILE_CHANNELS/HW_PE_ROWS-equivalent reference
-- point); if that case's shape ever changes, these must change with it.
entity tb_cnn_accel_pe_array_from_vectors is
  generic (
    -- VUnit's own per-test output directory, filled in by VUnit itself.
    -- module_cnn_accel.py's pre_config hook runs generate_vectors.
    -- generate_pe_array_xlang_case() into it right before the simulation
    -- starts, so the case is read from '<output_path>/pe_array_xlang_check'
    -- -- nothing is read from the repository, no vector is checked in.
    output_path : string;
    runner_cfg : string
  );
end entity tb_cnn_accel_pe_array_from_vectors;

architecture tb of tb_cnn_accel_pe_array_from_vectors is

  constant vectors_path : string := output_path & "/pe_array_xlang_check";

  -- Must match generate_vectors.py's pe_array_xlang_check case
  -- (HW_TILE_CHANNELS/HW_PE_ROWS) and desc.txt exactly -- see this file's
  -- header comment.
  constant c_pe_rows : positive := 8;
  constant c_pe_cols : positive := 8;
  constant c_tile_channels : positive := 8;
  constant c_accum_width : positive := 32;
  constant c_max_kernel : positive := 1;
  constant c_in_w : positive := 3;
  constant c_in_h : positive := 3;
  constant c_in_c : positive := 10;
  constant c_out_c : positive := 5;
  constant c_num_pixels : positive := c_in_w * c_in_h;
  constant c_num_tiles : positive := (c_in_c + c_tile_channels - 1) / c_tile_channels;
  constant c_weight_buffer_depth : positive := 4;

  constant c_window_len : positive := window_data_length(c_max_kernel, c_tile_channels);
  constant c_addr_width : positive := num_bits_needed(c_weight_buffer_depth - 1);
  constant c_weight_lanes : positive := c_pe_rows * c_pe_cols;

  constant c_clk_period : time := 10 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal cfg_kernel_h : std_ulogic_vector(7 downto 0) :=
    std_ulogic_vector(to_unsigned(c_max_kernel, 8));
  signal cfg_kernel_w : std_ulogic_vector(7 downto 0) :=
    std_ulogic_vector(to_unsigned(c_max_kernel, 8));

  signal s_window_m2s : window_m2s_t(data(0 to c_window_len - 1)) := (
    valid => '0', last => '0', first_tile => '0', last_tile => '0',
    data => (others => (others => '0'))
  );
  signal s_window_s2m : window_s2m_t;

  signal weight_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0);
  signal weight_rd_data : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0) := (others => '0');

  signal m_accum_m2s : accum_m2s_t(data(0 to c_pe_rows - 1)(c_accum_width - 1 downto 0));
  signal m_accum_s2m : accum_s2m_t := (ready => '0');

  -- Weight "memory" content, modelling cnn_accel_weight_buffer's own read
  -- port contract (registered, 1-cycle latency) -- same idiom as
  -- tb_cnn_accel_pe_array.vhd, but filled from weights_packed.txt (this
  -- file's whole point) instead of randomized.
  type weight_row_t is array (0 to c_weight_lanes - 1) of integer range -128 to 127;
  type weight_mem_t is array (0 to c_weight_buffer_depth - 1) of weight_row_t;
  signal weight_mem_s : weight_mem_t := (others => (others => 0));

  type window_row_t is array (0 to c_window_len - 1) of integer range -128 to 127;
  type accum_int_arr_t is array (0 to c_pe_rows - 1) of integer;

  -- Flat integer buffers read straight from the vector files (plain
  -- 'integer', unconstrained by the int8/int32 subtypes above -- narrowed
  -- on assignment into weight_mem_s/window_row_t/accum_int_arr_t, which
  -- range-checks each value).
  type flat_int_arr_t is array (natural range <>) of integer;

  constant c_num_weight_values : positive := c_num_tiles * c_weight_lanes;
  constant c_num_input_values : positive := c_num_pixels * c_in_c;
  constant c_num_accum_values : positive := c_num_pixels * c_pe_rows;

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  ------------------------------------------------------------------------
  -- Vector-file reader: one signed decimal integer per line, no header --
  -- exactly generate_vectors.py's own '_write_int_lines' format (see
  -- doc/cnn_accel_test_vectors.md).
  ------------------------------------------------------------------------

  procedure read_int_file(file_name : string; data : out flat_int_arr_t) is
    file f : text;
    variable l : line;
    variable v : integer;
    variable status : file_open_status;
  begin
    file_open(status, f, file_name, read_mode);
    assert status = open_ok
      report "tb_cnn_accel_pe_array_from_vectors: could not open '" & file_name & "'"
      severity failure;
    for i in data'range loop
      assert not endfile(f)
        report "tb_cnn_accel_pe_array_from_vectors: unexpected EOF in '" & file_name & "'"
        severity failure;
      readline(f, l);
      read(l, v);
      data(i) := v;
    end loop;
    file_close(f);
  end procedure;

  -- Packs one weight row (c_weight_lanes int8 lanes) into 'weight_rd_data's
  -- own bit layout (lane l at bits 8*(l+1)-1 downto 8*l) -- same layout as
  -- tb_cnn_accel_pe_array.vhd's identical helper.
  function pack_weight_row(row : weight_row_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);
  begin
    for i in 0 to c_weight_lanes - 1 loop
      result(8 * (i + 1) - 1 downto 8 * i) := std_ulogic_vector(to_signed(row(i), 8));
    end loop;
    return result;
  end function;

  -- Packs one pixel's expected per-row accumulator results, same lane
  -- layout as pack_actual below.
  function pack_expected(values : accum_int_arr_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(c_accum_width * c_pe_rows - 1 downto 0);
  begin
    for i in 0 to c_pe_rows - 1 loop
      result(c_accum_width * (i + 1) - 1 downto c_accum_width * i) :=
        std_ulogic_vector(to_signed(values(i), c_accum_width));
    end loop;
    return result;
  end function;

  -- Flattens the DUT's own 'm_accum_m2s.data' into the same flat layout as
  -- pack_expected, for a single check_equal comparison.
  function pack_actual(data : accum_array_t) return std_ulogic_vector is
    constant elem_width : positive := data(data'left)'length;
    variable result : std_ulogic_vector(data'length * elem_width - 1 downto 0);
    variable idx : natural := 0;
  begin
    for i in data'range loop
      result(elem_width * (idx + 1) - 1 downto elem_width * idx) := std_ulogic_vector(data(i));
      idx := idx + 1;
    end loop;
    return result;
  end function;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_pe_array
    generic map (
      g_pe_rows => c_pe_rows,
      g_pe_cols => c_pe_cols,
      g_accum_width => c_accum_width,
      g_max_kernel_size => c_max_kernel,
      g_tile_channels => c_tile_channels,
      g_weight_buffer_depth => c_weight_buffer_depth
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_kernel_h => cfg_kernel_h,
      cfg_kernel_w => cfg_kernel_w,

      s_window_m2s => s_window_m2s,
      s_window_s2m => s_window_s2m,

      weight_rd_addr => weight_rd_addr,
      weight_rd_data => weight_rd_data,

      m_accum_m2s => m_accum_m2s,
      m_accum_s2m => m_accum_s2m
    );

  ------------------------------------------------------------------------
  -- Weight "memory" model: registered, 1-cycle read latency -- same
  -- contract/idiom as tb_cnn_accel_pe_array.vhd's identical process.
  ------------------------------------------------------------------------

  weight_mem_model : process(clk)
    variable addr_int : natural;
  begin
    if rising_edge(clk) then
      addr_int := to_integer(unsigned(weight_rd_addr));
      if addr_int <= c_weight_buffer_depth - 1 then
        weight_rd_data <= pack_weight_row(weight_mem_s(addr_int));
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable weights_flat : flat_int_arr_t(0 to c_num_weight_values - 1);
    variable input_flat : flat_int_arr_t(0 to c_num_input_values - 1);
    variable accum_flat : flat_int_arr_t(0 to c_num_accum_values - 1);

    variable window : window_row_t;
    variable expected : accum_int_arr_t;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    -- Load this run's whole fixed input from the checked-in vector case
    -- (generate_vectors.py's pe_array_xlang_check) once, before any
    -- stimulus -- mirrors tb_cnn_accel_pe_array.vhd's own
    -- randomize_weight_mem-before-any-beat idiom.
    read_int_file(vectors_path & "/weights_packed.txt", weights_flat);
    read_int_file(vectors_path & "/input.txt", input_flat);
    read_int_file(vectors_path & "/pe_array_raw_accum.txt", accum_flat);

    for row in 0 to c_num_tiles - 1 loop
      for lane in 0 to c_weight_lanes - 1 loop
        weight_mem_s(row)(lane) <= weights_flat(row * c_weight_lanes + lane);
      end loop;
    end loop;

    do_reset;
    wait until rising_edge(clk);

    if run("test_cross_language_bitexact") then
      m_accum_s2m.ready <= '1';

      for pixel in 0 to c_num_pixels - 1 loop
        for tile in 0 to c_num_tiles - 1 loop
          -- Kernel is fixed 1x1 here, so one beat's window is exactly one
          -- input-channel tile's slice of this pixel's channel vector,
          -- zero-padded past c_in_c (D11) -- no spatial tap/padding logic
          -- needed (that is cnn_accel_window_gen's own concern, already
          -- covered by tb_cnn_accel_window_gen.vhd).
          for c in 0 to c_tile_channels - 1 loop
            if tile * c_tile_channels + c < c_in_c then
              window(c) := input_flat(pixel * c_in_c + tile * c_tile_channels + c);
            else
              window(c) := 0;
            end if;
          end loop;

          for i in 0 to c_window_len - 1 loop
            s_window_m2s.data(i) <= std_ulogic_vector(to_signed(window(i), 8));
          end loop;
          s_window_m2s.first_tile <= to_sl(tile = 0);
          s_window_m2s.last_tile <= to_sl(tile = c_num_tiles - 1);
          s_window_m2s.last <= to_sl(pixel = c_num_pixels - 1 and tile = c_num_tiles - 1);
          s_window_m2s.valid <= '1';
          wait until rising_edge(clk) and s_window_s2m.ready = '1';
          s_window_m2s.valid <= '0';
        end loop;

        for r in 0 to c_pe_rows - 1 loop
          expected(r) := accum_flat(pixel * c_pe_rows + r);
        end loop;

        wait until rising_edge(clk) and m_accum_m2s.valid = '1' and m_accum_s2m.ready = '1';
        check_equal(
          pack_actual(m_accum_m2s.data), pack_expected(expected),
          "m_accum data mismatch (Python-packed weights vs. real RTL) at pixel " &
          integer'image(pixel)
        );
        check_equal(
          m_accum_m2s.last, to_sl(pixel = c_num_pixels - 1),
          "m_accum last mismatch at pixel " & integer'image(pixel)
        );
      end loop;

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
