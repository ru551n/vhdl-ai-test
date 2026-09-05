library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- Configurable K_h x K_w / stride / zero-padding sliding-window generator.
-- See modules/cnn_accel/doc/cnn_accel_window_gen_req.md and
-- modules/cnn_accel/doc/cnn_accel_window_gen_proposal.md.
--
-- Architecture: 'g_max_kernel_size' full-row banks (BRAM-inference intent,
-- each 'g_max_fmap_width' pixels wide), ping-ponged by physical input row
-- number modulo 'g_max_kernel_size' -- literally the requirement doc's own
-- "buffers K_h - 1 full rows ... plus the current row ... ping-pong across
-- K_h row banks" description, not a FIFO/shift-register pipeline (see
-- proposal doc section 4 for why a FIFO-based design, closer to
-- modules/canny/src/canny_window3x3.vhd's fixed-3x3 technique, is
-- insufficient here: canny has no configurable stride/padding, so it never
-- needs to *replay* an already-fully-written row/column for more than one
-- output position; this module's padding can make a bottom/right output
-- row or column position's real (unpadded) row/column identical to an
-- earlier one's, which a pop-once FIFO or a shallow shift register can no
-- longer supply once its head has moved on, but a full-row bank -- read
-- (never popped) at whatever column address a given tap needs -- still
-- can).
entity cnn_accel_window_gen is
  generic (
    -- Upper bound on 'cfg_kernel_h'/'cfg_kernel_w'; sizes the row-bank
    -- count (below) and the window's tap grid. Contract:
    -- 'g_max_kernel_size >= 2' (asserted below) and
    -- 'g_max_kernel_size**2 * g_line_buffer_channels * 8 <= axi_stream_data_sz'
    -- (128), matching cnn_accel_pool's identical contract on its own
    -- 'g_max_kernel_size', since 'm_window_m2s.data' uses the fixed-width
    -- 'axi_stream_pkg' record type -- see proposal doc section 2.
    g_max_kernel_size : positive;
    -- Upper bound on 'cfg_in_width'; sizes each row bank's depth
    -- (BRAM-inference intent).
    g_max_fmap_width : positive;
    -- Channels processed in parallel per beat. Contract: 'cfg_in_channels'
    -- must equal this value (no channel tiling in v1 -- see proposal doc
    -- section 3).
    g_line_buffer_channels : positive
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
    -- Pulse: the final window of the frame has been accepted
    -- (m_window_s2m.ready = '1' the same cycle).
    done : out std_ulogic;
    --# {{}}
    -- Raster-order int8 (x g_line_buffer_channels) input pixels, from the
    -- ifmap cnn_accel_axi_read_dma.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- One K_h x K_w x channels window per beat. 'data' low
    -- 'cfg_kernel_h * cfg_kernel_w * g_line_buffer_channels * 8' bits hold
    -- the window: tap 'i' (row-major, 'i = row * cfg_kernel_w + col'),
    -- channel 'c', at bits '8*(i*g_line_buffer_channels + c) + 7 downto
    -- 8*(i*g_line_buffer_channels + c)'; remaining high bits are '0'.
    -- 'last' is '1' only for the final window of the final output row.
    m_window_m2s : out axi_stream_m2s_t;
    m_window_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_window_gen;

architecture a of cnn_accel_window_gen is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_lane_width : positive := 8 * g_line_buffer_channels;
  constant c_window_data_width : positive :=
    g_max_kernel_size * g_max_kernel_size * c_lane_width;

  ------------------------------------------------------------------------
  -- Row banks: 'g_max_kernel_size' full-row buffers (BRAM-inference
  -- intent), each 'g_max_fmap_width' pixels wide. Physical input row
  -- number 'r' always lives in bank 'r mod g_max_kernel_size'; since
  -- 'cfg_kernel_h <= g_max_kernel_size' (asserted at 'start'), at most
  -- 'g_max_kernel_size' distinct physical rows (the current row plus up
  -- to 'g_max_kernel_size - 1' previous ones) are ever simultaneously
  -- needed by any pending/future output row, so this many banks never
  -- forces an unread row to be evicted -- see the entity-level comment
  -- above and proposal doc section 4.
  --
  -- Unlike a FIFO (popped once, in strict order) or a shallow column-tap
  -- shift register (only the last few columns of a row), a bank is a
  -- plain read/write array: writes go to 'row_banks(cur_row mod
  -- g_max_kernel_size)(cur_col)' as pixels arrive, and 'assemble_window'
  -- below reads 'row_banks(input_row mod g_max_kernel_size)(input_col)'
  -- directly, for whichever '(input_row, input_col)' a given tap needs,
  -- however many output positions end up needing that same already-
  -- written cell (padding-induced replay, see entity-level comment).
  ------------------------------------------------------------------------

  type row_bank_t is array (0 to g_max_fmap_width - 1) of
    std_ulogic_vector(c_lane_width - 1 downto 0);
  type row_bank_arr_t is array (0 to g_max_kernel_size - 1) of row_bank_t;

  signal row_banks : row_bank_arr_t;

  signal fire : std_ulogic;
  signal window_valid : std_ulogic;
  signal consume : std_ulogic;
  signal last_window : std_ulogic;

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

  ------------------------------------------------------------------------
  -- Position counters.
  --
  -- 'cur_row_q'/'cur_col_q': the input pixel position currently being
  -- written (row/col within the raw, unpadded frame).
  -- 'out_row_q'/'out_col_q': the next output window position to produce.
  ------------------------------------------------------------------------

  signal cur_row_q, cur_col_q : unsigned(15 downto 0) := (others => '0');
  signal out_row_q, out_col_q : unsigned(15 downto 0) := (others => '0');

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

  assert c_window_data_width <= axi_stream_data_sz
    report "cnn_accel_window_gen: g_max_kernel_size**2 * g_line_buffer_channels * 8 " &
      "must be <= axi_stream_data_sz (128)"
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
  begin
    if rising_edge(clk) then
      if reset = '1' then
        active_q <= '0';
        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');

      elsif start = '1' then
        assert unsigned(cfg_kernel_h) <= to_unsigned(g_max_kernel_size, 8)
          and unsigned(cfg_kernel_w) <= to_unsigned(g_max_kernel_size, 8)
          report "cnn_accel_window_gen: cfg_kernel_h/w must be <= g_max_kernel_size"
          severity failure;
        assert unsigned(cfg_in_channels) = to_unsigned(g_line_buffer_channels, 16)
          report "cnn_accel_window_gen: cfg_in_channels must equal g_line_buffer_channels " &
            "(no channel tiling in v1)"
          severity warning;

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
        -- integer division here is deliberate -- see proposal doc
        -- section 5.
        v_num_w := to_integer(unsigned(cfg_in_width)) + to_integer(unsigned(cfg_pad_left))
          + to_integer(unsigned(cfg_pad_right)) - to_integer(unsigned(cfg_kernel_w));
        v_num_h := to_integer(unsigned(cfg_in_height)) + to_integer(unsigned(cfg_pad_top))
          + to_integer(unsigned(cfg_pad_bottom)) - to_integer(unsigned(cfg_kernel_h));

        out_width_q <= to_unsigned(v_num_w / to_integer(unsigned(cfg_stride_w)) + 1, 16);
        out_height_q <= to_unsigned(v_num_h / to_integer(unsigned(cfg_stride_h)) + 1, 16);

        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        active_q <= '1';

      else
        if fire = '1' then
          row_banks(to_integer(cur_row_q) mod g_max_kernel_size)(to_integer(cur_col_q)) <=
            s_stream_m2s.data(c_lane_width - 1 downto 0);

          if cur_col_q = in_width_q - 1 then
            cur_col_q <= (others => '0');
            cur_row_q <= cur_row_q + 1;
          else
            cur_col_q <= cur_col_q + 1;
          end if;
        end if;

        if consume = '1' then
          if out_col_q = out_width_q - 1 then
            out_col_q <= (others => '0');
            if out_row_q /= out_height_q - 1 then
              out_row_q <= out_row_q + 1;
            end if;
          else
            out_col_q <= out_col_q + 1;
          end if;

          if last_window = '1' then
            active_q <= '0';
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  last_window <= '1' when (out_row_q = out_height_q - 1 and out_col_q = out_width_q - 1) else '0';
  done <= consume and last_window;

  m_window_m2s.valid <= window_valid;
  m_window_m2s.last <= last_window;
  m_window_m2s.user <= (others => '0');

  ------------------------------------------------------------------------
  -- Window readiness + assembly. 'window_valid' is a pure function of
  -- registered state (never of 's_stream_m2s.valid'/'m_window_s2m.ready'),
  -- per proposal doc section 4/Axi4.md.
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
    row_banks, active_q
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
    -- (or no real column at all) -- see proposal doc section 4.
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
            -- row 'input_row' at column 'input_col' -- no pop/shift
            -- ordering constraint, so this is correct however many
            -- output positions end up reading the same already-written
            -- cell (see the row-bank comment above).
            bank_idx := input_row mod g_max_kernel_size;
            tap := row_banks(bank_idx)(input_col);

            tap_idx := kr * kw + kc;
            data_i(c_lane_width * (tap_idx + 1) - 1 downto c_lane_width * tap_idx) := tap;
          end if;
        end if;
      end loop;
    end loop;

    m_window_m2s.data <= (others => '0');
    m_window_m2s.data(c_window_data_width - 1 downto 0) <= data_i;
  end process;

end architecture a;
