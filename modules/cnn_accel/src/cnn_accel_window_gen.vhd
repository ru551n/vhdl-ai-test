library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Configurable K_h x K_w / stride / zero-padding / input-channel-tiled
-- sliding-window generator. See modules/cnn_accel/doc/cnn_accel_window_gen_req.md,
-- doc/cnn_accel_window_gen_proposal.md (pre-tiling design) and
-- doc/cnn_accel_tiled_dataflow_proposal.md sections 1/2/9 (the channel-
-- tiling retrofit implemented here).
--
-- Architecture: 'g_max_kernel_size' full-row banks (BRAM-inference intent,
-- each 'g_max_row_tile_words' *channel-tile cells* wide -- 'in_width *
-- ceil(in_channels/g_tile_channels)' cells, not one cell per column, see
-- below), ping-ponged by physical input row number modulo
-- 'g_max_kernel_size' -- literally the requirement doc's own "buffers
-- K_h - 1 full rows ... plus the current row ... ping-pong across K_h row
-- banks" description, not a FIFO/shift-register pipeline (see
-- cnn_accel_window_gen_proposal.md section 4 for why a FIFO-based design,
-- closer to modules/canny/src/canny_window3x3.vhd's fixed-3x3 technique,
-- is insufficient here: canny has no configurable stride/padding, so it
-- never needs to *replay* an already-fully-written row/column for more
-- than one output position; this module's padding can make a
-- bottom/right output row or column position's real (unpadded)
-- row/column identical to an earlier one's, which a pop-once FIFO or a
-- shallow shift register can no longer supply once its head has moved
-- on, but a full-row bank -- read (never popped) at whatever address a
-- given tap needs -- still can).
--
-- Channel tiling (cnn_accel_tiled_dataflow_proposal.md sections 1/2):
-- 'cfg_in_channels' need not equal 'g_tile_channels'. Each output pixel
-- emits 'T = ceil(cfg_in_channels / g_tile_channels)' consecutive
-- 'm_window' beats, one per input-channel tile, with 'first_tile'/
-- 'last_tile' sidebands marking the first/last tile of that pixel (both
-- '1' when 'T = 1'). The stream-level 'last' flag still means "last
-- output pixel of the whole feature map" and is asserted only on the
-- final beat ('last_tile' = '1') of the final pixel -- never on every
-- 'last_tile'. Symmetrically, one accepted 's_stream' beat now writes one
-- '(col, tile)' cell (not one full pixel): 'T' beats/pixel on ingest too.
-- When 'cfg_in_channels' is not a multiple of 'g_tile_channels', the
-- final tile's unused channel lanes are driven to '0' at write time
-- (D11) -- deterministically, not left as whatever garbage
-- 's_stream_m2s.data' happens to carry above the valid byte range.
entity cnn_accel_window_gen is
  generic (
    -- Upper bound on 'cfg_kernel_h'/'cfg_kernel_w'; sizes the row-bank
    -- count (below) and the window's tap grid. Contract:
    -- 'g_max_kernel_size >= 2' (asserted below).
    g_max_kernel_size : positive;
    -- Upper bound on 'cfg_in_width * ceil(cfg_in_channels / g_tile_channels)'
    -- ("row-tile-word count"); sizes each row bank's depth (BRAM-inference
    -- intent). Bounding the *product* (rather than sizing width and
    -- channel-count independently) is deliberate -- see
    -- cnn_accel_tiled_dataflow_proposal.md section 1. Checked by a
    -- 'severity failure' assert at 'start' (runtime values, not a true
    -- generic-only elaboration bound).
    g_max_row_tile_words : positive;
    -- Input channels processed in parallel per beat/tile ('Ct'). Contract:
    -- 'g_tile_channels * 8 <= axi_stream_data_sz' (asserted below), since
    -- one tile's channels must fit in 's_stream_m2s.data''s low bytes.
    -- 'cfg_in_channels' need not be a multiple of this -- see the
    -- entity-level comment on channel tiling / D11 zero-padding.
    g_tile_channels : positive
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- Kernel/stride/padding/frame-size configuration, latched at 'start'.
    cfg_kernel_h : in std_ulogic_vector(7 downto 0);
    cfg_kernel_w : in std_ulogic_vector(7 downto 0);
    cfg_stride_h : in std_ulogic_vector(7 downto 0);
    cfg_stride_w : in std_ulogic_vector(7 downto 0);
    cfg_pad_top : in std_ulogic_vector(7 downto 0);
    cfg_pad_bottom : in std_ulogic_vector(7 downto 0);
    cfg_pad_left : in std_ulogic_vector(7 downto 0);
    cfg_pad_right : in std_ulogic_vector(7 downto 0);
    cfg_in_width : in std_ulogic_vector(15 downto 0);
    cfg_in_height : in std_ulogic_vector(15 downto 0);
    cfg_in_channels : in std_ulogic_vector(15 downto 0);
    --# {{}}
    -- Pulse, from cnn_accel_layer_ctrl: latches the 'cfg_*' ports above and
    -- resets row/column counters and line-buffer pointers for a new frame.
    start : in std_ulogic;
    -- Pulse: the final tile beat of the final window of the frame has
    -- been accepted (m_window_s2m.ready = '1' the same cycle).
    done : out std_ulogic;
    --# {{}}
    -- Raster-order int8 input pixels, from the ifmap
    -- cnn_accel_axi_read_dma. One accepted beat writes one channel-tile
    -- of one column; 'data' low '8 * g_tile_channels' bits hold that
    -- tile's channels (remaining high bits, if any, are ignored) -- see
    -- the entity-level comment on channel tiling.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- One K_h x K_w x g_tile_channels window per beat, 'T' consecutive
    -- beats per output pixel (one per input-channel tile). 'data' low
    -- 'cfg_kernel_h * cfg_kernel_w * g_tile_channels * 8' bits hold the
    -- window: tap 'i' (row-major, 'i = row * cfg_kernel_w + col'), channel
    -- 'c' within the tile, at bits '8*(i*g_tile_channels + c) + 7 downto
    -- 8*(i*g_tile_channels + c)'; remaining high bits are '0'. Partial
    -- final-tile lanes ('c' beyond the valid channel count) read as '0'
    -- (D11), not garbage. 'first_tile'/'last_tile' mark the first/last
    -- tile of the current output pixel (both '1' when 'T = 1'). 'last' is
    -- '1' only for the final tile beat of the final window of the frame.
    m_window_m2s : out window_m2s_t(data(window_data_width(g_max_kernel_size, g_tile_channels) - 1 downto 0));
    m_window_s2m : in window_s2m_t
  );
end entity cnn_accel_window_gen;

architecture a of cnn_accel_window_gen is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_lane_width : positive := 8 * g_tile_channels;
  constant c_window_data_width : positive := window_data_width(g_max_kernel_size, g_tile_channels);

  ------------------------------------------------------------------------
  -- Row banks: 'g_max_kernel_size' full-row buffers (BRAM-inference
  -- intent), each 'g_max_row_tile_words' channel-tile cells wide -- cell
  -- 'col * n_tiles + tile' holds column 'col''s tile 'tile' (one
  -- channel-tile of int8 activations, 'c_lane_width' bits), not one cell
  -- per column -- see the entity-level comment on channel tiling and
  -- cnn_accel_tiled_dataflow_proposal.md section 1. Physical input row
  -- number 'r' always lives in bank 'r mod g_max_kernel_size'; since
  -- 'cfg_kernel_h <= g_max_kernel_size' (asserted at 'start'), at most
  -- 'g_max_kernel_size' distinct physical rows (the current row plus up
  -- to 'g_max_kernel_size - 1' previous ones) are ever simultaneously
  -- needed by any pending/future output row, so this many banks never
  -- forces an unread row to be evicted -- see the entity-level comment
  -- above and cnn_accel_window_gen_proposal.md section 4.
  --
  -- Unlike a FIFO (popped once, in strict order) or a shallow column-tap
  -- shift register (only the last few columns of a row), a bank is a
  -- plain read/write array: writes go to 'row_banks(cur_row mod
  -- g_max_kernel_size)(cur_col * n_tiles + wr_tile)' as pixel-tiles
  -- arrive, and 'assemble_window' below reads 'row_banks(input_row mod
  -- g_max_kernel_size)(input_col * n_tiles + rd_tile)' directly, for
  -- whichever '(input_row, input_col, tile)' a given tap needs, however
  -- many output positions end up needing that same already-written cell
  -- (padding-induced replay, see entity-level comment).
  ------------------------------------------------------------------------

  type row_bank_t is array (0 to g_max_row_tile_words - 1) of
    std_ulogic_vector(c_lane_width - 1 downto 0);
  type row_bank_arr_t is array (0 to g_max_kernel_size - 1) of row_bank_t;

  signal row_banks : row_bank_arr_t;

  signal fire : std_ulogic;
  signal window_valid : std_ulogic;
  signal consume : std_ulogic;
  signal last_pixel : std_ulogic;
  signal first_tile_flag, last_tile_flag : std_ulogic;

  -- '1' once a frame is in progress (between 'start' and the final
  -- window's acceptance); gates 's_stream_s2m.ready'/'window_valid' so
  -- nothing is accepted/emitted before the first 'start'.
  signal active_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Configuration, latched at 'start'.
  ------------------------------------------------------------------------

  signal kernel_h_q, kernel_w_q : unsigned(7 downto 0) := (others => '0');
  signal stride_h_q, stride_w_q : unsigned(7 downto 0) := (others => '0');
  signal pad_top_q, pad_left_q : unsigned(7 downto 0) := (others => '0');
  signal in_width_q, in_height_q : unsigned(15 downto 0) := (others => '0');
  signal out_width_q, out_height_q : unsigned(15 downto 0) := (others => '0');

  -- 'T = ceil(cfg_in_channels / g_tile_channels)', the number of
  -- input-channel tiles per output pixel (and per ingested input pixel).
  signal n_tiles_q : unsigned(15 downto 0) := to_unsigned(1, 16);
  -- Number of valid (real, non-zero-padded) channels in the *last* tile
  -- of the frame; equals 'g_tile_channels' when 'cfg_in_channels' is an
  -- exact multiple of it, 'cfg_in_channels mod g_tile_channels'
  -- otherwise. Only the last tile of a frame can ever be partial (a
  -- direct consequence of 'T' being a ceiling division) -- see D11 in
  -- the entity-level comment.
  signal last_tile_channels_q : natural range 0 to g_tile_channels := g_tile_channels;

  ------------------------------------------------------------------------
  -- Position counters.
  --
  -- 'cur_row_q'/'cur_col_q': the input pixel position currently being
  -- written (row/col within the raw, unpadded frame).
  -- 'wr_tile_q': the input-channel tile currently being written, within
  -- 'cur_col_q' ('0 .. n_tiles_q - 1').
  -- 'out_row_q'/'out_col_q': the next output window position to produce.
  -- 'rd_tile_q': the input-channel tile currently being read/emitted,
  -- within 'out_row_q'/'out_col_q' ('0 .. n_tiles_q - 1').
  ------------------------------------------------------------------------

  signal cur_row_q, cur_col_q : unsigned(15 downto 0) := (others => '0');
  signal wr_tile_q : unsigned(15 downto 0) := (others => '0');
  signal out_row_q, out_col_q : unsigned(15 downto 0) := (others => '0');
  signal rd_tile_q : unsigned(15 downto 0) := (others => '0');

  function imin(a, b : integer) return integer is
  begin
    if a < b then
      return a;
    else
      return b;
    end if;
  end function;

begin

  assert g_max_kernel_size >= 2
    report "cnn_accel_window_gen: g_max_kernel_size must be >= 2"
    severity failure;

  assert g_tile_channels * 8 <= axi_stream_data_sz
    report "cnn_accel_window_gen: g_tile_channels * 8 must be <= axi_stream_data_sz " &
      "(128) -- one channel-tile must fit in one s_stream beat"
    severity failure;

  ------------------------------------------------------------------------
  fire <= s_stream_m2s.valid and s_stream_s2m.ready;
  consume <= window_valid and m_window_s2m.ready;

  -- Freeze further input acceptance whenever a window is pending and not
  -- yet consumed (a row bank must not be overwritten until every tap that
  -- still needs it has been read out). 'window_valid' depends only on
  -- registered state (never on 's_stream_m2s.valid'/'m_window_s2m.ready'),
  -- so this has no combinational loop.
  s_stream_s2m.ready <= active_q and (not window_valid or m_window_s2m.ready);

  ------------------------------------------------------------------------
  -- Configuration latch, position counters, and row-bank writes.
  ------------------------------------------------------------------------
  control : process(clk)
    variable v_num_w, v_num_h : integer;
    variable v_in_channels, v_n_tiles : integer;
    variable v_write_data : std_ulogic_vector(c_lane_width - 1 downto 0);
  begin
    if rising_edge(clk) then
      if reset = '1' then
        active_q <= '0';
        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        wr_tile_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        rd_tile_q <= (others => '0');

      elsif start = '1' then
        assert unsigned(cfg_kernel_h) <= to_unsigned(g_max_kernel_size, 8)
          and unsigned(cfg_kernel_w) <= to_unsigned(g_max_kernel_size, 8)
          report "cnn_accel_window_gen: cfg_kernel_h/w must be <= g_max_kernel_size"
          severity failure;

        kernel_h_q <= unsigned(cfg_kernel_h);
        kernel_w_q <= unsigned(cfg_kernel_w);
        stride_h_q <= unsigned(cfg_stride_h);
        stride_w_q <= unsigned(cfg_stride_w);
        pad_top_q <= unsigned(cfg_pad_top);
        pad_left_q <= unsigned(cfg_pad_left);
        in_width_q <= unsigned(cfg_in_width);
        in_height_q <= unsigned(cfg_in_height);

        -- out_dim = (in_dim + pad_lo + pad_hi - kernel) / stride + 1.
        -- One-shot per 'start' (not a per-cycle datapath), so plain
        -- integer division here is deliberate -- see
        -- cnn_accel_window_gen_proposal.md section 5.
        v_num_w := to_integer(unsigned(cfg_in_width)) + to_integer(unsigned(cfg_pad_left))
          + to_integer(unsigned(cfg_pad_right)) - to_integer(unsigned(cfg_kernel_w));
        v_num_h := to_integer(unsigned(cfg_in_height)) + to_integer(unsigned(cfg_pad_top))
          + to_integer(unsigned(cfg_pad_bottom)) - to_integer(unsigned(cfg_kernel_h));

        out_width_q <= to_unsigned(v_num_w / to_integer(unsigned(cfg_stride_w)) + 1, 16);
        out_height_q <= to_unsigned(v_num_h / to_integer(unsigned(cfg_stride_h)) + 1, 16);

        -- T = ceil(in_channels / g_tile_channels); only the last tile of
        -- the frame can be partial (see 'last_tile_channels_q's comment).
        v_in_channels := to_integer(unsigned(cfg_in_channels));
        v_n_tiles := (v_in_channels + g_tile_channels - 1) / g_tile_channels;

        assert to_integer(unsigned(cfg_in_width)) * v_n_tiles <= g_max_row_tile_words
          report "cnn_accel_window_gen: cfg_in_width * ceil(cfg_in_channels/g_tile_channels) " &
            "must be <= g_max_row_tile_words"
          severity failure;

        n_tiles_q <= to_unsigned(v_n_tiles, 16);
        last_tile_channels_q <= v_in_channels - (v_n_tiles - 1) * g_tile_channels;

        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        wr_tile_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        rd_tile_q <= (others => '0');
        active_q <= '1';

      else
        if fire = '1' then
          -- D11: zero-pad the unused high channel lanes of a partial
          -- final tile, deterministically -- not whatever
          -- 's_stream_m2s.data' happens to carry there. Non-final tiles,
          -- and an exactly-dividing final tile ('last_tile_channels_q =
          -- g_tile_channels', making this a null loop range), are
          -- written verbatim.
          -- The loop range is constant (0 .. g_tile_channels - 1) with the
          -- runtime bound applied as a per-lane condition inside, rather
          -- than the more direct 'for c in last_tile_channels_q to ...'.
          -- Both are identical in simulation, but a variable loop range is
          -- not synthesizable: it crashes GHDL's synthesis backend
          -- ("limits of range are not constant", then an Ada assertion in
          -- synth-vhdl_expr.adb). Caught by this module's netlist build --
          -- see module_cnn_accel.py get_build_projects().
          v_write_data := s_stream_m2s.data(c_lane_width - 1 downto 0);
          if wr_tile_q = n_tiles_q - 1 then
            for c in 0 to g_tile_channels - 1 loop
              if c >= last_tile_channels_q then
                v_write_data(8 * (c + 1) - 1 downto 8 * c) := (others => '0');
              end if;
            end loop;
          end if;

          row_banks(to_integer(cur_row_q) mod g_max_kernel_size)(
            to_integer(cur_col_q) * to_integer(n_tiles_q) + to_integer(wr_tile_q)
          ) <= v_write_data;

          if wr_tile_q = n_tiles_q - 1 then
            wr_tile_q <= (others => '0');
            if cur_col_q = in_width_q - 1 then
              cur_col_q <= (others => '0');
              cur_row_q <= cur_row_q + 1;
            else
              cur_col_q <= cur_col_q + 1;
            end if;
          else
            wr_tile_q <= wr_tile_q + 1;
          end if;
        end if;

        if consume = '1' then
          if rd_tile_q = n_tiles_q - 1 then
            rd_tile_q <= (others => '0');

            if out_col_q = out_width_q - 1 then
              out_col_q <= (others => '0');
              if out_row_q /= out_height_q - 1 then
                out_row_q <= out_row_q + 1;
              end if;
            else
              out_col_q <= out_col_q + 1;
            end if;

            if last_pixel = '1' then
              active_q <= '0';
            end if;
          else
            rd_tile_q <= rd_tile_q + 1;
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  last_pixel <= '1' when (out_row_q = out_height_q - 1 and out_col_q = out_width_q - 1) else '0';
  first_tile_flag <= '1' when rd_tile_q = 0 else '0';
  last_tile_flag <= '1' when rd_tile_q = n_tiles_q - 1 else '0';
  done <= consume and last_pixel and last_tile_flag;

  m_window_m2s.valid <= window_valid;
  m_window_m2s.last <= last_pixel and last_tile_flag;
  m_window_m2s.first_tile <= first_tile_flag;
  m_window_m2s.last_tile <= last_tile_flag;

  ------------------------------------------------------------------------
  -- Window readiness + assembly. 'window_valid' is a pure function of
  -- registered state (never of 's_stream_m2s.valid'/'m_window_s2m.ready'),
  -- per cnn_accel_window_gen_proposal.md section 4/Axi4.md. The
  -- 'row_ready' spatial test is channel/tile-agnostic (unchanged from the
  -- pre-tiling design) -- every tile of a given output pixel reads the
  -- same row/column range, tiling only changes which row-bank cell
  -- ('input_col * n_tiles_q + rd_tile_q') supplies each tap -- see
  -- cnn_accel_tiled_dataflow_proposal.md section 2's orthogonality
  -- argument.
  --
  -- NOTE: explicit sensitivity list, not 'process(all)' -- see
  -- canny_window3x3.vhd's identical note on GHDL 7.0.0-dev's 'all'
  -- inference not reliably tracking signals read only through nested
  -- loops/array indexing.
  ------------------------------------------------------------------------
  assemble_window : process(
    kernel_h_q, kernel_w_q, stride_h_q, stride_w_q, pad_top_q, pad_left_q,
    in_width_q, in_height_q, cur_row_q, cur_col_q,
    out_row_q, out_col_q, out_width_q, out_height_q,
    row_banks, active_q, n_tiles_q, rd_tile_q
  )
    variable kh, kw, sh, sw, pt, pl, inh, inw : integer;
    variable orow, ocol : integer;
    variable row_top, row_bot, real_row_bot : integer;
    variable col_left, col_right, real_col_right : integer;
    variable has_real_row, has_real_col : boolean;
    variable row_ready : boolean;
    variable input_row, input_col : integer;
    variable in_frame : boolean;
    variable data_i : std_ulogic_vector(c_window_data_width - 1 downto 0);
    variable tap : std_ulogic_vector(c_lane_width - 1 downto 0);
    variable tap_idx : integer;
    variable bank_idx : integer;
    variable n_tiles_i, rd_tile_i : integer;
  begin
    kh := to_integer(kernel_h_q);
    kw := to_integer(kernel_w_q);
    sh := to_integer(stride_h_q);
    sw := to_integer(stride_w_q);
    pt := to_integer(pad_top_q);
    pl := to_integer(pad_left_q);
    inh := to_integer(in_height_q);
    inw := to_integer(in_width_q);
    orow := to_integer(out_row_q);
    ocol := to_integer(out_col_q);
    n_tiles_i := to_integer(n_tiles_q);
    rd_tile_i := to_integer(rd_tile_q);

    -- Row range needed by this window: [row_top, row_bot] (may extend
    -- outside [0, inh-1) -- top/bottom padding).
    row_top := orow * sh - pt;
    row_bot := row_top + kh - 1;
    has_real_row := (row_top <= inh - 1) and (row_bot >= 0);
    real_row_bot := imin(row_bot, inh - 1);

    -- Column range needed by this window: [col_left, col_right].
    col_left := ocol * sw - pl;
    col_right := col_left + kw - 1;
    has_real_col := (col_left <= inw - 1) and (col_right >= 0);
    real_col_right := imin(col_right, inw - 1);

    -- Ready once the largest real row/column this window needs has been
    -- fully written: no real row at all (window entirely vertical
    -- padding) -> ready immediately; the needed row already complete
    -- ('cur_row_q > real_row_bot') -> ready regardless of columns;
    -- still writing that exact row -> also need its columns caught up
    -- (or no real column at all) -- see cnn_accel_window_gen_proposal.md
    -- section 4.
    row_ready :=
      (not has_real_row)
      or (to_integer(cur_row_q) > real_row_bot)
      or (
        to_integer(cur_row_q) = real_row_bot
        and ((not has_real_col) or (to_integer(cur_col_q) > real_col_right))
      );

    window_valid <= '1' when (active_q = '1' and row_ready) else '0';

    data_i := (others => '0');

    for kr in 0 to g_max_kernel_size - 1 loop
      for kc in 0 to g_max_kernel_size - 1 loop
        if kr < kh and kc < kw then
          input_row := orow * sh + kr - pt;
          input_col := ocol * sw + kc - pl;
          in_frame := input_row >= 0 and input_row <= inh - 1
            and input_col >= 0 and input_col <= inw - 1;

          if in_frame then
            -- Direct random-access read of the row bank holding physical
            -- row 'input_row', channel-tile 'rd_tile_q' of column
            -- 'input_col' -- no pop/shift ordering constraint, so this
            -- is correct however many output positions end up reading
            -- the same already-written cell (see the row-bank comment
            -- above). Partial-final-tile zero-padding (D11) already
            -- happened at write time, so no extra masking is needed here.
            bank_idx := input_row mod g_max_kernel_size;
            tap := row_banks(bank_idx)(input_col * n_tiles_i + rd_tile_i);

            -- 'tap_idx' depends on the runtime 'kw', so writing
            -- 'data_i(c_lane_width*(tap_idx+1)-1 downto c_lane_width*tap_idx)'
            -- directly is a dynamic slice, which crashes GHDL's synthesis
            -- backend (Ada assertion in synth-vhdl_expr.adb). Select the
            -- destination tap slot with a constant-bound loop instead --
            -- identical in simulation, a mux in hardware. 'tap_idx' can
            -- only ever land in 0 .. g_max_kernel_size**2 - 1, since
            -- 'kr, kc < kh, kw <= g_max_kernel_size'.
            tap_idx := kr * kw + kc;
            for t in 0 to g_max_kernel_size * g_max_kernel_size - 1 loop
              if t = tap_idx then
                data_i(c_lane_width * (t + 1) - 1 downto c_lane_width * t) := tap;
              end if;
            end loop;
          end if;
        end if;
      end loop;
    end loop;

    m_window_m2s.data <= (others => '0');
    m_window_m2s.data(c_window_data_width - 1 downto 0) <= data_i;
  end process;

end architecture a;
