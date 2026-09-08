library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.queue_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- VUnit-5 testbench for cnn_accel_window_gen, covering both the
-- pre-tiling spatial state machine (kernel/stride/padding/backpressure,
-- unchanged since the channel-tiling retrofit is orthogonal to it -- see
-- doc/cnn_accel_tiled_dataflow_proposal.md section 2/9) and the
-- input-channel-tiling retrofit itself (T=1/partial/T>1/exact/partial,
-- first_tile/last_tile sequencing, D11 zero-padding of unused final-tile
-- lanes).
--
-- Golden model: an independent full-frame array ('frame_t') indexed
-- directly by (row, col, channel), with out-of-'[0, in_dim)'/out-of-
-- '[0, in_channels)' lookups returning 0 (spatial padding / D11 channel
-- padding respectively) -- deliberately *not* the RTL's own row-bank/
-- tile-counter machinery, so a bug in that machinery cannot also be
-- present in the oracle. Hand-rolled record-port stimulus/monitor
-- procedures (no VUnit axi_stream_master/slave VC), mirroring
-- tb_cnn_accel_weight_buffer.vhd/tb_cnn_accel_pool.vhd's identical choice
-- for record-typed ports.
entity tb_cnn_accel_window_gen is
  generic (runner_cfg : string);
end entity tb_cnn_accel_window_gen;

architecture tb of tb_cnn_accel_window_gen is

  -- Small, directed generics. 'c_kernel_max = 3' (not 2): a max kernel
  -- size of 2 would make the column-tap shift chain's 'for k in
  -- g_max_kernel_size-1 downto 2' loop an empty range (0 iterations),
  -- never exercising a shift depth beyond 1 -- see
  -- doc/cnn_accel_window_gen_proposal.md's "Verification plan".
  -- 'c_tile_channels = 8' mirrors the design proposal's recommended
  -- 'g_tile_channels = g_pe_cols' default (doc/cnn_accel_tiled_dataflow_
  -- proposal.md section 1/3).
  constant c_kernel_max : positive := 3;
  constant c_fmap_max : positive := 9;
  constant c_tile_channels : positive := 8;
  -- Largest 'cfg_in_channels' exercised by any test below (T>1 partial).
  constant c_channels_max : positive := 20;
  -- Bound on 'in_width * ceil(in_channels/g_tile_channels)'; largest
  -- product actually exercised below is 12 (6 cols * T=2, and 4 cols *
  -- T=3) -- sized with headroom.
  constant c_row_tile_words_max : positive := 32;

  constant c_max_window_length : positive := window_data_length(c_kernel_max, c_tile_channels);
  -- Flat-bit width of that many int8 elements. Only used for the VUnit
  -- queue (push/pop have no overload for a user-defined array type) and
  -- for check_equal's diff output; the DUT port itself is the array.
  constant c_max_window_bits : positive := 8 * c_max_window_length;
  constant c_frame_max : positive := c_fmap_max;

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
  -- ISA v2.1: the int8 value padded taps take (the tensor's zero-point).
  signal cfg_pad_value : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_in_width : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_in_height : std_ulogic_vector(15 downto 0) := (others => '0');
  signal cfg_in_channels : std_ulogic_vector(15 downto 0) := (others => '0');

  signal start : std_ulogic := '0';
  signal done : std_ulogic;

  signal s_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_stream_s2m : axi_stream_s2m_t;

  signal m_window_m2s : window_m2s_t(data(0 to c_max_window_length - 1));
  signal m_window_s2m : window_s2m_t := (ready => '0');

  -- Monitor's current randomized-stall probability (percent); the main
  -- process sets this before streaming each frame -- mirrors
  -- tb_cnn_accel_pool.vhd's per-link stall generics, but as a plain
  -- signal (no per-test generic sweep support in module_cnn_accel.py yet).
  signal cur_stall_out : natural := 0;

  -- Reset synchronously clears this; counts 'done' pulses since the last
  -- reset, so 'run_frame' can assert "exactly one 'done' per frame" --
  -- equivalently "'last' asserted exactly once, on the final beat", since
  -- 'done <= consume and m_window_m2s.last' in the DUT.
  signal done_pulse_count : natural := 0;

  -- Non-blocking scoreboard: (window data, first_tile, last_tile, last)
  -- tuples, in output raster x tile order, enqueued up-front by
  -- 'enqueue_frame_expected' before the frame's pixels are even streamed
  -- in (output order is a pure function of the configuration, independent
  -- of pacing) and popped/checked by 'monitor' on every accepted
  -- 'm_window' beat.
  constant expected_q : queue_t := new_queue;

  type frame_t is array (0 to c_frame_max - 1, 0 to c_frame_max - 1, 0 to c_channels_max - 1)
    of integer range -128 to 127;

  type shape_t is record
    h : natural;
    w : natural;
  end record;
  type shape_arr_t is array (natural range <>) of shape_t;

  -- Kernel shapes swept by test_kernel_stride_shapes: 1x1 up to
  -- c_kernel_max x c_kernel_max, square and non-square (mirrors
  -- tb_cnn_accel_pool.vhd's identical 'c_shapes').
  constant c_shapes : shape_arr_t(0 to 4) := (
    (1, 1), (2, 2), (3, 3), (2, 3), (3, 2)
  );
  constant c_strides : shape_arr_t(0 to 1) := (
    (1, 1), (2, 2)
  );

  type pad_t is record
    top : natural;
    bottom : natural;
    left : natural;
    right : natural;
  end record;
  type pad_arr_t is array (natural range <>) of pad_t;

  -- Padding combinations swept by test_padding_all_sides: none, symmetric,
  -- and two asymmetric combinations (pad_top /= pad_bottom, pad_left /=
  -- pad_right -- including one large enough, relative to the 3x3 kernel,
  -- to clip a window's real-row range down to a single row).
  constant c_pads : pad_arr_t(0 to 3) := (
    (0, 0, 0, 0), (1, 1, 1, 1), (2, 0, 0, 2), (1, 2, 2, 1)
  );

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  -- T = ceil(in_channels / c_tile_channels), the number of input-channel
  -- tiles per output pixel -- mirrors the DUT's own 'n_tiles_q'.
  function n_tiles(in_channels : natural) return natural is
  begin
    return (in_channels + c_tile_channels - 1) / c_tile_channels;
  end function;

  ------------------------------------------------------------------------
  -- Golden model.
  ------------------------------------------------------------------------

  -- Returns 'pad_value' for any (row, col) outside
  -- '[0, in_h) x [0, in_w)' (spatial padding, ISA v2.1: the pad fill is
  -- the tensor's zero-point, not necessarily 0) and 0 for any 'channel'
  -- outside '[0, in_channels)' (D11's final-tile-lane zero-padding --
  -- deliberately still 0: that is a *channel* lane that does not exist,
  -- not a spatial position outside the frame), the real pixel otherwise.
  function golden_tap(
    frame : frame_t; in_h, in_w, in_channels : natural; row, col, channel : integer;
    pad_value : integer
  ) return integer is
  begin
    if row < 0 or col < 0 or row > in_h - 1 or col > in_w - 1 then
      return pad_value;
    end if;
    if channel > in_channels - 1 then
      return 0;
    end if;
    return frame(row, col, channel);
  end function;

  -- Packs one (output pixel, tile) beat's taps (row-major,
  -- 'tap_idx = kr * kw + kc'; channel 'c' within the tile) into the low
  -- 'kh * kw * c_tile_channels * 8' bits of a fixed 'c_max_window_bits'-
  -- wide vector, per cnn_accel_pkg's documented 'window_m2s_t.data'
  -- layout; remaining high bits stay 0.
  function golden_window(
    frame : frame_t;
    in_h, in_w, in_channels, kh, kw, sh, sw, pad_top, pad_left : natural;
    out_row, out_col, tile_idx : natural;
    pad_value : integer
  ) return std_ulogic_vector is
    -- Every element starts at the pad value, not 0: the DUT clears its
    -- whole tap-assembly register to 'cfg_pad_value' at the start of each
    -- window, so the taps beyond the runtime 'kh * kw' (which no consumer
    -- looks at) hold the pad value as well.
    variable result : std_ulogic_vector(c_max_window_bits - 1 downto 0) := (others => '0');
    variable input_row, input_col, tap_idx, abs_channel : integer;
  begin
    for i in 0 to c_max_window_length - 1 loop
      result(8 * (i + 1) - 1 downto 8 * i) := std_ulogic_vector(to_signed(pad_value, 8));
    end loop;
    for kr in 0 to kh - 1 loop
      for kc in 0 to kw - 1 loop
        input_row := out_row * sh + kr - pad_top;
        input_col := out_col * sw + kc - pad_left;
        tap_idx := kr * kw + kc;
        for c in 0 to c_tile_channels - 1 loop
          abs_channel := tile_idx * c_tile_channels + c;
          result(8 * (tap_idx * c_tile_channels + c + 1) - 1 downto 8 * (tap_idx * c_tile_channels + c)) :=
            std_ulogic_vector(to_signed(
              golden_tap(
                frame, in_h, in_w, in_channels, input_row, input_col, abs_channel, pad_value
              ), 8
            ));
        end loop;
      end loop;
    end loop;
    return result;
  end function;

begin

  clk <= not clk after c_clk_period / 2;

  ------------------------------------------------------------------------
  -- Structural safety net (whole-simulation, not just one dedicated
  -- test): 'done' must coincide exactly with the acceptance of the final
  -- ('last') window beat -- see cnn_accel_window_gen.vhd's
  -- 'done <= consume and last_pixel and last_tile_flag'.
  ------------------------------------------------------------------------

  done_relation_check : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        done_pulse_count <= 0;
      else
        if done = '1' then
          check_true(
            m_window_m2s.valid = '1' and m_window_s2m.ready = '1' and m_window_m2s.last = '1'
              and m_window_m2s.last_tile = '1',
            "done must coincide with acceptance of the final ('last', 'last_tile') window beat"
          );
          done_pulse_count <= done_pulse_count + 1;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  dut : entity cnn_accel.cnn_accel_window_gen
    generic map (
      g_max_kernel_size => c_kernel_max,
      g_max_row_tile_words => c_row_tile_words_max,
      g_tile_channels => c_tile_channels
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
      cfg_in_channels => cfg_in_channels,
      start => start,
      done => done,
      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,
      m_window_m2s => m_window_m2s,
      m_window_s2m => m_window_s2m
    );

  ------------------------------------------------------------------------
  -- Output monitor: independent randomized-'ready' responder. Pops
  -- +checks 'expected_q' on every accepted 'm_window' beat -- data,
  -- 'first_tile', 'last_tile' and 'last' all cross-checked against the
  -- golden model, so first_tile/last_tile sequencing errors (e.g. a
  -- missing/duplicated tile beat, a flag raised on the wrong beat) show
  -- up as ordinary scoreboard mismatches across every test, not just a
  -- dedicated one.
  ------------------------------------------------------------------------

  monitor : process
    variable rnd : RandomPType;
    variable expected_data : std_ulogic_vector(c_max_window_bits - 1 downto 0);
    variable expected_first_tile, expected_last_tile, expected_last : std_ulogic;
  begin
    rnd.InitSeed(get_string_seed(runner_cfg) & "_window_monitor");
    m_window_s2m.ready <= '0';
    wait until reset = '0' and rising_edge(clk);

    loop
      if rnd.RandInt(0, 99) < cur_stall_out then
        m_window_s2m.ready <= '0';
        for i in 1 to rnd.RandInt(1, 3) loop
          wait until rising_edge(clk);
        end loop;
      end if;

      m_window_s2m.ready <= '1';
      wait until rising_edge(clk);

      if m_window_m2s.valid = '1' and m_window_s2m.ready = '1' then
        expected_data := pop(expected_q);
        expected_first_tile := pop(expected_q);
        expected_last_tile := pop(expected_q);
        expected_last := pop(expected_q);
        check_equal(
          to_slv(m_window_m2s.data), expected_data, "m_window data mismatch"
        );
        check_equal(m_window_m2s.first_tile, expected_first_tile, "m_window first_tile mismatch");
        check_equal(m_window_m2s.last_tile, expected_last_tile, "m_window last_tile mismatch");
        check_equal(m_window_m2s.last, expected_last, "m_window last mismatch");
      end if;
    end loop;
  end process;

  ------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;
    variable frame : frame_t;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
    end procedure;

    procedure random_frame(in_h, in_w : natural) is
    begin
      for r in 0 to in_h - 1 loop
        for c in 0 to in_w - 1 loop
          for ch in 0 to c_channels_max - 1 loop
            frame(r, c, ch) := rnd.RandInt(-128, 127);
          end loop;
        end loop;
      end loop;
    end procedure;

    -- Pushes one 's_stream' beat carrying tile 'tile_idx' of pixel
    -- (r, c) (one of 'n_tiles(in_channels)' beats for that pixel), with
    -- randomized input-side stall. Channels at/above 'in_channels' within
    -- this tile (only possible in the final tile -- D11) are filled with
    -- deliberately *non-zero* garbage, not 0: the point is to prove the
    -- DUT's own write-side zeroing (not a coincidentally-zero source)
    -- is what makes those lanes read back as 0.
    procedure push_tile(
      r, c : natural; in_channels, tile_idx : natural; stall_percent : natural
    ) is
      variable data : std_ulogic_vector(axi_stream_data_sz - 1 downto 0) := (others => '0');
      variable abs_channel : integer;
    begin
      for lane in 0 to c_tile_channels - 1 loop
        abs_channel := tile_idx * c_tile_channels + lane;
        if abs_channel <= in_channels - 1 then
          data(8 * (lane + 1) - 1 downto 8 * lane) :=
            std_ulogic_vector(to_signed(frame(r, c, abs_channel), 8));
        else
          data(8 * (lane + 1) - 1 downto 8 * lane) :=
            std_ulogic_vector(to_signed(rnd.RandInt(1, 127), 8));
        end if;
      end loop;
      s_stream_m2s.data <= data;
      if rnd.RandInt(0, 99) < stall_percent then
        s_stream_m2s.valid <= '0';
        for i in 1 to rnd.RandInt(1, 3) loop
          wait until rising_edge(clk);
        end loop;
      end if;
      s_stream_m2s.valid <= '1';
      wait until rising_edge(clk) and s_stream_s2m.ready = '1';
      s_stream_m2s.valid <= '0';
    end procedure;

    -- Streams rows '0 .. in_h - 2' at full 'in_w' width, then row 'in_h - 1'
    -- (the last row being streamed at all) only up to column
    -- 'last_col_needed' (inclusive; a negative value streams no columns on
    -- that row) -- see 'run_frame''s comment on why the very last streamed
    -- row can also have unread *trailing columns*, not just unread
    -- trailing rows, whenever stride does not evenly divide the padded
    -- frame width. A null column range ('last_col_needed < 0') is a
    -- correct, ordinary empty 'for' loop in VHDL, not a special case.
    -- Every pixel actually streamed pushes 'n_tiles(in_channels)'
    -- consecutive tile beats, mirroring the DUT's own write-side tiling.
    procedure stream_frame(
      in_h, in_w, in_channels : natural; stall_percent : natural; last_col_needed : integer
    ) is
      variable col_limit : integer;
    begin
      for r in 0 to in_h - 1 loop
        if r = in_h - 1 then
          col_limit := last_col_needed;
        else
          col_limit := in_w - 1;
        end if;
        for c in 0 to col_limit loop
          for tile in 0 to n_tiles(in_channels) - 1 loop
            push_tile(r, c, in_channels, tile, stall_percent);
          end loop;
        end loop;
      end loop;
    end procedure;

    -- Latches the 'cfg_*' configuration and pulses 'start'; the DUT
    -- samples 'cfg_*' the same edge 'start' is high, so these must be
    -- assigned before (not after) the pulse.
    procedure begin_frame(
      kh, kw, sh, sw, pt, pb, pl, pr, inw, inh, in_channels : natural;
      pad_value : integer := 0
    ) is
    begin
      cfg_pad_value <= std_ulogic_vector(to_signed(pad_value, 8));
      cfg_kernel_h <= std_ulogic_vector(to_unsigned(kh, 8));
      cfg_kernel_w <= std_ulogic_vector(to_unsigned(kw, 8));
      cfg_stride_h <= std_ulogic_vector(to_unsigned(sh, 8));
      cfg_stride_w <= std_ulogic_vector(to_unsigned(sw, 8));
      cfg_pad_top <= std_ulogic_vector(to_unsigned(pt, 8));
      cfg_pad_bottom <= std_ulogic_vector(to_unsigned(pb, 8));
      cfg_pad_left <= std_ulogic_vector(to_unsigned(pl, 8));
      cfg_pad_right <= std_ulogic_vector(to_unsigned(pr, 8));
      cfg_in_width <= std_ulogic_vector(to_unsigned(inw, 16));
      cfg_in_height <= std_ulogic_vector(to_unsigned(inh, 16));
      cfg_in_channels <= std_ulogic_vector(to_unsigned(in_channels, 16));
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
    end procedure;

    -- Enqueues every expected output window beat (in raster x tile order)
    -- for a frame already latched via 'begin_frame' -- see 'golden_window'
    -- above. 'first_tile'/'last_tile' are both '1' on the single beat
    -- when 'T = 1'; 'last' fires only on the last tile beat of the final
    -- pixel.
    procedure enqueue_frame_expected(
      in_h, in_w, in_channels, kh, kw, sh, sw, pad_top, pad_left, out_h, out_w : natural;
      pad_value : integer := 0
    ) is
      variable t : natural;
    begin
      t := n_tiles(in_channels);
      for orow in 0 to out_h - 1 loop
        for ocol in 0 to out_w - 1 loop
          for tile in 0 to t - 1 loop
            push(expected_q, golden_window(
              frame, in_h, in_w, in_channels, kh, kw, sh, sw, pad_top, pad_left, orow, ocol,
              tile, pad_value
            ));
            push(expected_q, to_sl(tile = 0));
            push(expected_q, to_sl(tile = t - 1));
            push(expected_q, to_sl(
              orow = out_h - 1 and ocol = out_w - 1 and tile = t - 1
            ));
          end loop;
        end loop;
      end loop;
    end procedure;

    -- Waits (bounded) until 'expected_q' has drained, then confirms it is
    -- truly empty (every expected window beat actually arrived).
    procedure drain_and_check(max_wait_cycles : positive) is
      variable cycles : natural := 0;
    begin
      while not is_empty(expected_q) and cycles < max_wait_cycles loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
      end loop;
      check_true(is_empty(expected_q), "m_window scoreboard queue did not drain in time");
    end procedure;

    -- One full frame end-to-end: reset, latch config + start, enqueue the
    -- golden expected window beats, stream the frame's pixel-tile beats
    -- (honoring 'stall_in'), drain the scoreboard, and confirm exactly
    -- one 'done' pulse fired.
    procedure run_frame(
      in_h, in_w, kh, kw, sh, sw, pt, pb, pl, pr : natural;
      stall_in, stall_out : natural;
      max_wait_cycles : positive;
      in_channels : natural := c_tile_channels;
      pad_value : integer := 0
    ) is
      variable out_w, out_h : natural;
      variable last_row_needed, last_col_needed : integer;
      variable rows_to_stream : natural;
    begin
      do_reset;
      begin_frame(kh, kw, sh, sw, pt, pb, pl, pr, in_w, in_h, in_channels, pad_value);
      out_w := (in_w + pl + pr - kw) / sw + 1;
      out_h := (in_h + pt + pb - kh) / sh + 1;
      enqueue_frame_expected(
        in_h, in_w, in_channels, kh, kw, sh, sw, pt, pl, out_h, out_w, pad_value
      );
      cur_stall_out <= stall_out;
      -- The DUT can legitimately finish a frame (deassert
      -- 's_stream_s2m.ready' via its own 'active_q', per 'done's contract
      -- of firing once the final *output* window beat is accepted) before
      -- every raw input pixel-tile has arrived, whenever 'stride' does
      -- not evenly divide '(in_h + pad_top + pad_bottom - kh)' /
      -- '(in_w + pad_left + pad_right - kw)' -- the last output
      -- row/column's real, unpadded bottom row/rightmost column
      -- ('last_row_needed'/'last_col_needed' below, mirroring
      -- cnn_accel_window_gen.vhd's own 'row_ready' formula) can then
      -- leave one or more trailing input rows, *and/or trailing columns
      -- of the very last streamed row*, genuinely unread. Streaming
      -- those never-consumed trailing rows/columns would block
      -- 'push_tile' forever waiting for a 's_stream_s2m.ready' that will
      -- never come again this frame -- discovered the hard way: an
      -- earlier revision only trimmed whole trailing rows, which still
      -- deadlocked on a stride/kernel combination (e.g. kh=kw=3,
      -- sh=sw=2, in_h=7, in_w=8) whose *last needed row* is the frame's
      -- actual last row (no whole row is skippable) but whose last
      -- needed *column* on that row is still short of 'in_w - 1'.
      last_row_needed := (out_h - 1) * sh - pt + kh - 1;
      if last_row_needed > in_h - 1 then
        last_row_needed := in_h - 1;
      end if;
      rows_to_stream := last_row_needed + 1;
      last_col_needed := (out_w - 1) * sw - pl + kw - 1;
      if last_col_needed > in_w - 1 then
        last_col_needed := in_w - 1;
      end if;
      stream_frame(rows_to_stream, in_w, in_channels, stall_in, last_col_needed);
      drain_and_check(max_wait_cycles);
      -- Settle: 'done_pulse_count' is updated by a separate clocked
      -- process off the very same edge 'drain_and_check''s loop just woke
      -- from; without this, reading it here risks a delta-cycle race
      -- (seeing the pre-increment value even though this edge's 'done'
      -- pulse already logically happened).
      wait for 1 ns;
      check_equal(done_pulse_count, 1, "exactly one 'done' pulse per frame");
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    if run("test_kernel_stride_shapes") then
      for s in c_shapes'range loop
        for st in c_strides'range loop
          random_frame(7, 8);
          run_frame(
            7, 8, c_shapes(s).h, c_shapes(s).w, c_strides(st).h, c_strides(st).w,
            0, 0, 0, 0, 15, 15, 2000
          );
        end loop;
      end loop;

    elsif run("test_padding_all_sides") then
      for p in c_pads'range loop
        random_frame(5, 5);
        run_frame(
          5, 5, 3, 3, 1, 1,
          c_pads(p).top, c_pads(p).bottom, c_pads(p).left, c_pads(p).right,
          15, 15, 2000
        );
      end loop;

    elsif run("test_pad_value") then
      -- ISA v2.1: padded taps take 'cfg_pad_value', not 0. Swept over the
      -- interesting int8 values -- 0 (the pre-v2.1 behaviour, which must
      -- be bit-identical to what test_padding_all_sides already checks),
      -- -128 (YOLOv8n's activation zero-point, the case this exists for),
      -- +127 and an arbitrary positive -- across every padding
      -- combination, so both the spatial pad taps and the
      -- beyond-the-kernel taps are checked against the golden model.
      for p in c_pads'range loop
        random_frame(5, 5);
        run_frame(
          5, 5, 3, 3, 1, 1,
          c_pads(p).top, c_pads(p).bottom, c_pads(p).left, c_pads(p).right,
          15, 15, 2000, c_tile_channels, -128
        );

        random_frame(5, 5);
        run_frame(
          5, 5, 3, 3, 1, 1,
          c_pads(p).top, c_pads(p).bottom, c_pads(p).left, c_pads(p).right,
          15, 15, 2000, c_tile_channels, 127
        );

        random_frame(5, 5);
        run_frame(
          5, 5, 3, 3, 1, 1,
          c_pads(p).top, c_pads(p).bottom, c_pads(p).left, c_pads(p).right,
          15, 15, 2000, c_tile_channels, 37
        );
      end loop;

      -- A pad value combined with channel tiling: the D11 unused-lane
      -- fill stays 0 (it is a nonexistent channel, not a spatial pad),
      -- while the spatial pad taps take -128. Getting these two confused
      -- is the obvious way to implement this wrong.
      random_frame(4, 4);
      run_frame(4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 30, 30, 2000, 3, -128);

      -- Stride 2 with a pad value: the pad taps of the *bottom/right*
      -- windows, which only exist because padding extends the frame.
      random_frame(5, 5);
      run_frame(5, 5, 3, 3, 2, 2, 1, 1, 1, 1, 20, 20, 2000, c_tile_channels, -128);

    elsif run("test_backpressure") then
      random_frame(6, 6);
      run_frame(6, 6, 2, 2, 1, 1, 0, 0, 0, 0, 40, 40, 3000);

      random_frame(6, 6);
      run_frame(6, 6, 3, 3, 2, 2, 1, 1, 1, 1, 40, 40, 3000);

      -- Mid-tile-sequence backpressure (T = 2): both 's_stream' and
      -- 'm_window' stall randomly *between individual tile beats of the
      -- same pixel*, not just between pixels -- must not lose or
      -- duplicate a beat (the scoreboard would catch either: a lost beat
      -- desyncs every subsequent 'first_tile'/'last_tile'/'data' check,
      -- a duplicated one leaves 'expected_q' non-empty at drain time).
      random_frame(6, 6);
      run_frame(6, 6, 3, 3, 1, 1, 1, 1, 1, 1, 40, 40, 4000, 2 * c_tile_channels);

    elsif run("test_full_throughput") then
      random_frame(8, 7);
      run_frame(8, 7, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 500);

    elsif run("test_channel_tiling_partial_single") then
      -- T = 1, partial: in_channels = 3 < c_tile_channels (8). The 5
      -- unused lanes (channels 3..7) must read back as 0 (D11) even
      -- though the source deliberately drives non-zero garbage there
      -- ('push_tile').
      random_frame(4, 4);
      run_frame(4, 4, 3, 3, 1, 1, 0, 0, 0, 0, 30, 30, 1000, 3);

    elsif run("test_channel_tiling_exact_multi") then
      -- T > 1, exact: in_channels = 16 = 2 * c_tile_channels. Every tile
      -- is full (no D11 zero-padding); exercises first_tile/last_tile
      -- sequencing (both tiles of every pixel across a whole small
      -- feature map) with stride 2 and asymmetric padding together.
      random_frame(5, 5);
      run_frame(5, 5, 3, 3, 2, 2, 1, 0, 0, 1, 30, 30, 2000, 2 * c_tile_channels);

    elsif run("test_channel_tiling_partial_multi") then
      -- T > 1, partial: in_channels = 20 = 2*c_tile_channels + 4, so
      -- T = 3 and only the last (3rd) tile is partial (4 valid channels,
      -- 4 unused D11-zeroed lanes). Exercises first_tile/last_tile
      -- sequencing across 3 tiles/pixel over a whole small feature map,
      -- combined with randomized backpressure on both links.
      random_frame(4, 4);
      run_frame(4, 4, 3, 3, 1, 1, 0, 0, 0, 0, 40, 40, 2000, 2 * c_tile_channels + 4);

    elsif run("test_reset_mid_frame_abort") then
      do_reset;
      -- Start a frame, stream only a fraction of its pixel-tiles, then
      -- abort via 'reset' mid-stream (no expectations were ever enqueued
      -- for this aborted session, since it must never be allowed to
      -- complete/be checked).
      begin_frame(3, 3, 1, 1, 0, 0, 0, 0, 6, 6, c_tile_channels);
      cur_stall_out <= 0;
      for i in 0 to 9 loop
        push_tile(0, 0, c_tile_channels, 0, 0);
      end loop;

      reset <= '1';
      wait until rising_edge(clk);
      reset <= '0';
      s_stream_m2s.valid <= '0';
      wait for 1 ns;
      check_true(s_stream_s2m.ready = '0', "s_stream_s2m.ready deasserted right after an abort (no frame active)");
      check_true(m_window_m2s.valid = '0', "m_window_m2s.valid deasserted right after an abort");
      check_equal(done, '0', "no 'done' pulse right after an abort");

      -- A fresh frame afterwards must behave exactly like any other
      -- fresh frame -- no leftover row/column/tile-counter state from the
      -- aborted session leaking in.
      random_frame(6, 6);
      run_frame(6, 6, 3, 3, 1, 1, 0, 0, 0, 0, 20, 20, 1000);

    end if;

    test_runner_cleanup(runner);
  end process;

  test_runner_watchdog(runner, 5 ms);

end architecture tb;
