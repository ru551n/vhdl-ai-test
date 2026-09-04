library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library canny_window3x3;
library canny_gaussian3x3;
library canny_sobel3x3;
library axi_stream_join;
library canny_nms;
library canny_threshold;
library canny_hysteresis;

-- Structural top: flat AXI4-Stream boundary ports, wiring the full Canny
-- pipeline per doc/canny_arch.md's block diagram and inter-module interface
-- table. No behavior of its own beyond structural wiring and the
-- flat<->record pack/unpack needed only for the reused hdl-modules
-- axi_stream_fifo instance (every other internal link is a flat
-- std_logic_vector AXI4-Stream link, matching the actual port style of
-- every submodule in this project -- see modules/canny_top/doc/canny_top_req.md
-- and the note there on this deviating from the arch doc's original
-- "everything is a record" framing, which none of the project's own new
-- modules ended up following).
entity canny_top is
  generic (
    g_img_width   : positive;
    g_img_height  : positive;
    g_thresh_low  : natural;
    g_thresh_high : natural
  );
  port (
    clk   : in std_logic;
    rst_n : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(7 downto 0);
    s_axis_tlast  : in  std_logic;
    s_axis_tuser  : in  std_logic_vector(0 downto 0);

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(7 downto 0);
    m_axis_tlast  : out std_logic;
    m_axis_tuser  : out std_logic_vector(0 downto 0)
  );
end entity canny_top;

architecture rtl of canny_top is

  -- W1 (raw pixel) -> gaussian
  signal w1_m_tvalid, w1_m_tready, w1_m_tlast : std_logic;
  signal w1_m_tdata : std_logic_vector(9 * 8 - 1 downto 0);
  signal w1_m_tuser : std_logic_vector(1 downto 0);

  -- gaussian -> W2
  signal gauss_m_tvalid, gauss_m_tready, gauss_m_tlast : std_logic;
  signal gauss_m_tdata : std_logic_vector(7 downto 0);
  signal gauss_m_tuser : std_logic_vector(1 downto 0);

  -- W2 (smoothed pixel) -> sobel
  signal w2_m_tvalid, w2_m_tready, w2_m_tlast : std_logic;
  signal w2_m_tdata : std_logic_vector(9 * 8 - 1 downto 0);
  signal w2_m_tuser : std_logic_vector(1 downto 0);

  -- sobel -> W3 (magnitude fork)
  signal sobel_mag_tvalid, sobel_mag_tready, sobel_mag_tlast : std_logic;
  signal sobel_mag_tdata : std_logic_vector(10 downto 0);
  signal sobel_mag_tuser : std_logic_vector(1 downto 0);

  -- sobel -> axi_stream_fifo (direction fork)
  signal sobel_dir_tvalid, sobel_dir_tready, sobel_dir_tlast : std_logic;
  signal sobel_dir_tdata : std_logic_vector(1 downto 0);
  signal sobel_dir_tuser : std_logic_vector(1 downto 0);

  -- W3 (magnitude window) -> join (a)
  signal w3_m_tvalid, w3_m_tready, w3_m_tlast : std_logic;
  signal w3_m_tdata : std_logic_vector(9 * 11 - 1 downto 0);
  signal w3_m_tuser : std_logic_vector(1 downto 0);

  -- axi_stream_fifo (hdl-modules, record-based) -- direction elastic buffer
  signal fifo_in_m2s  : axi_stream_m2s_t := axi_stream_m2s_init;
  signal fifo_in_s2m  : axi_stream_s2m_t := axi_stream_s2m_init;
  signal fifo_out_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal fifo_out_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  -- axi_stream_fifo output -> join (b)
  signal fifo_dir_tvalid, fifo_dir_tready, fifo_dir_tlast : std_logic;
  signal fifo_dir_tdata : std_logic_vector(1 downto 0);
  signal fifo_dir_tuser : std_logic_vector(1 downto 0);

  -- join -> nms
  signal join_m_tvalid, join_m_tready, join_m_tlast : std_logic;
  signal join_m_tdata : std_logic_vector(9 * 11 + 2 - 1 downto 0);
  signal join_m_tuser : std_logic_vector(1 downto 0);

  -- nms -> threshold
  signal nms_m_tvalid, nms_m_tready, nms_m_tlast : std_logic;
  signal nms_m_tdata : std_logic_vector(10 downto 0);
  signal nms_m_tuser : std_logic_vector(1 downto 0);

  -- threshold -> W4
  signal thresh_m_tvalid, thresh_m_tready, thresh_m_tlast : std_logic;
  signal thresh_m_tdata : std_logic_vector(1 downto 0);
  signal thresh_m_tuser : std_logic_vector(1 downto 0);

  -- W4 (classification window) -> hysteresis
  signal w4_m_tvalid, w4_m_tready, w4_m_tlast : std_logic;
  signal w4_m_tdata : std_logic_vector(9 * 2 - 1 downto 0);
  signal w4_m_tuser : std_logic_vector(1 downto 0);

  -- Direction-fork FIFO depth: must absorb the magnitude fork's extra
  -- window-fill latency (W3's ~2*g_img_width+2 cycles, per
  -- modules/canny_top/doc/canny_top_req.md) without ever back-pressuring
  -- sobel's direction output. Sized generously as a vhfill-time decision
  -- (not fixed by the arch doc); revisit if simulation/synthesis shows
  -- under- or over-provisioning. hdl-modules' fifo.vhd (that
  -- axi_stream_fifo/fifo_wrapper wrap) requires a power-of-two depth
  -- ("RAM depth must be a power of two"), so round the minimum
  -- requirement up to the next power of two.
  function next_pow2(minimum : positive) return positive is
    variable result : positive := 1;
  begin
    while result < minimum loop
      result := result * 2;
    end loop;
    return result;
  end function;

  constant c_dir_fifo_depth : positive := next_pow2(2 * g_img_width + 16);

begin

  ------------------------------------------------------------------------------
  w1_inst : entity canny_window3x3.canny_window3x3
    generic map (
      g_img_width  => g_img_width,
      g_img_height => g_img_height,
      g_data_width => 8,
      g_user_width => 1
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => s_axis_tvalid,
      s_axis_tready => s_axis_tready,
      s_axis_tdata  => s_axis_tdata,
      s_axis_tuser  => s_axis_tuser,
      s_axis_tlast  => s_axis_tlast,

      m_axis_tvalid => w1_m_tvalid,
      m_axis_tready => w1_m_tready,
      m_axis_tdata  => w1_m_tdata,
      m_axis_tuser  => w1_m_tuser,
      m_axis_tlast  => w1_m_tlast
    );

  ------------------------------------------------------------------------------
  gaussian_inst : entity canny_gaussian3x3.canny_gaussian3x3
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => w1_m_tvalid,
      s_axis_tready => w1_m_tready,
      s_axis_tdata  => w1_m_tdata,
      s_axis_tuser  => w1_m_tuser,
      s_axis_tlast  => w1_m_tlast,

      m_axis_tvalid => gauss_m_tvalid,
      m_axis_tready => gauss_m_tready,
      m_axis_tdata  => gauss_m_tdata,
      m_axis_tuser  => gauss_m_tuser,
      m_axis_tlast  => gauss_m_tlast
    );

  ------------------------------------------------------------------------------
  w2_inst : entity canny_window3x3.canny_window3x3
    generic map (
      g_img_width  => g_img_width,
      g_img_height => g_img_height,
      g_data_width => 8,
      g_user_width => 2
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => gauss_m_tvalid,
      s_axis_tready => gauss_m_tready,
      s_axis_tdata  => gauss_m_tdata,
      s_axis_tuser  => gauss_m_tuser,
      s_axis_tlast  => gauss_m_tlast,

      m_axis_tvalid => w2_m_tvalid,
      m_axis_tready => w2_m_tready,
      m_axis_tdata  => w2_m_tdata,
      m_axis_tuser  => w2_m_tuser,
      m_axis_tlast  => w2_m_tlast
    );

  ------------------------------------------------------------------------------
  sobel_inst : entity canny_sobel3x3.canny_sobel3x3
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => w2_m_tvalid,
      s_axis_tready => w2_m_tready,
      s_axis_tdata  => w2_m_tdata,
      s_axis_tuser  => w2_m_tuser,
      s_axis_tlast  => w2_m_tlast,

      m_axis_mag_tvalid => sobel_mag_tvalid,
      m_axis_mag_tready => sobel_mag_tready,
      m_axis_mag_tdata  => sobel_mag_tdata,
      m_axis_mag_tuser  => sobel_mag_tuser,
      m_axis_mag_tlast  => sobel_mag_tlast,

      m_axis_dir_tvalid => sobel_dir_tvalid,
      m_axis_dir_tready => sobel_dir_tready,
      m_axis_dir_tdata  => sobel_dir_tdata,
      m_axis_dir_tuser  => sobel_dir_tuser,
      m_axis_dir_tlast  => sobel_dir_tlast
    );

  ------------------------------------------------------------------------------
  w3_inst : entity canny_window3x3.canny_window3x3
    generic map (
      g_img_width  => g_img_width,
      g_img_height => g_img_height,
      g_data_width => 11,
      g_user_width => 2
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => sobel_mag_tvalid,
      s_axis_tready => sobel_mag_tready,
      s_axis_tdata  => sobel_mag_tdata,
      s_axis_tuser  => sobel_mag_tuser,
      s_axis_tlast  => sobel_mag_tlast,

      m_axis_tvalid => w3_m_tvalid,
      m_axis_tready => w3_m_tready,
      m_axis_tdata  => w3_m_tdata,
      m_axis_tuser  => w3_m_tuser,
      m_axis_tlast  => w3_m_tlast
    );

  ------------------------------------------------------------------------------
  -- Direction-fork elasticity: reused unmodified from hdl-modules. Its
  -- ports are the axi_stream_m2s_t/s2m_t record pair (not this project's
  -- flat convention), so a thin pack/unpack sits directly around this
  -- instance only -- see the entity-level comment above.
  dir_fifo_pack : process (all)
  begin
    fifo_in_m2s        <= axi_stream_m2s_init;
    fifo_in_m2s.valid  <= sobel_dir_tvalid;
    fifo_in_m2s.data(sobel_dir_tdata'range) <= sobel_dir_tdata;
    fifo_in_m2s.last   <= sobel_dir_tlast;
    fifo_in_m2s.user(sobel_dir_tuser'range) <= sobel_dir_tuser;
  end process;

  sobel_dir_tready <= fifo_in_s2m.ready;

  fifo_dir_tvalid <= fifo_out_m2s.valid;
  fifo_dir_tdata  <= fifo_out_m2s.data(fifo_dir_tdata'range);
  fifo_dir_tlast  <= fifo_out_m2s.last;
  fifo_dir_tuser  <= fifo_out_m2s.user(fifo_dir_tuser'range);

  fifo_out_s2m.ready <= fifo_dir_tready;

  dir_fifo_inst : entity axi_stream.axi_stream_fifo
    generic map (
      data_width   => 2,
      user_width   => 2,
      asynchronous => false,
      depth        => c_dir_fifo_depth
    )
    port map (
      clk => clk,

      input_m2s => fifo_in_m2s,
      input_s2m => fifo_in_s2m,

      output_m2s => fifo_out_m2s,
      output_s2m => fifo_out_s2m
    );

  ------------------------------------------------------------------------------
  join_inst : entity axi_stream_join.axi_stream_join
    generic map (
      g_data_width_a => 9 * 11,
      g_data_width_b => 2
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_a_tvalid => w3_m_tvalid,
      s_axis_a_tready => w3_m_tready,
      s_axis_a_tdata  => w3_m_tdata,
      s_axis_a_tuser  => w3_m_tuser,
      s_axis_a_tlast  => w3_m_tlast,

      s_axis_b_tvalid => fifo_dir_tvalid,
      s_axis_b_tready => fifo_dir_tready,
      s_axis_b_tdata  => fifo_dir_tdata,
      s_axis_b_tuser  => fifo_dir_tuser,
      s_axis_b_tlast  => fifo_dir_tlast,

      m_axis_tvalid => join_m_tvalid,
      m_axis_tready => join_m_tready,
      m_axis_tdata  => join_m_tdata,
      m_axis_tuser  => join_m_tuser,
      m_axis_tlast  => join_m_tlast
    );

  ------------------------------------------------------------------------------
  nms_inst : entity canny_nms.canny_nms
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => join_m_tvalid,
      s_axis_tready => join_m_tready,
      s_axis_tdata  => join_m_tdata,
      s_axis_tuser  => join_m_tuser,
      s_axis_tlast  => join_m_tlast,

      m_axis_tvalid => nms_m_tvalid,
      m_axis_tready => nms_m_tready,
      m_axis_tdata  => nms_m_tdata,
      m_axis_tuser  => nms_m_tuser,
      m_axis_tlast  => nms_m_tlast
    );

  ------------------------------------------------------------------------------
  threshold_inst : entity canny_threshold.canny_threshold
    generic map (
      g_thresh_low  => g_thresh_low,
      g_thresh_high => g_thresh_high
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => nms_m_tvalid,
      s_axis_tready => nms_m_tready,
      s_axis_tdata  => nms_m_tdata,
      s_axis_tuser  => nms_m_tuser,
      s_axis_tlast  => nms_m_tlast,

      m_axis_tvalid => thresh_m_tvalid,
      m_axis_tready => thresh_m_tready,
      m_axis_tdata  => thresh_m_tdata,
      m_axis_tuser  => thresh_m_tuser,
      m_axis_tlast  => thresh_m_tlast
    );

  ------------------------------------------------------------------------------
  w4_inst : entity canny_window3x3.canny_window3x3
    generic map (
      g_img_width  => g_img_width,
      g_img_height => g_img_height,
      g_data_width => 2,
      g_user_width => 2
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => thresh_m_tvalid,
      s_axis_tready => thresh_m_tready,
      s_axis_tdata  => thresh_m_tdata,
      s_axis_tuser  => thresh_m_tuser,
      s_axis_tlast  => thresh_m_tlast,

      m_axis_tvalid => w4_m_tvalid,
      m_axis_tready => w4_m_tready,
      m_axis_tdata  => w4_m_tdata,
      m_axis_tuser  => w4_m_tuser,
      m_axis_tlast  => w4_m_tlast
    );

  ------------------------------------------------------------------------------
  hysteresis_inst : entity canny_hysteresis.canny_hysteresis
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => w4_m_tvalid,
      s_axis_tready => w4_m_tready,
      s_axis_tdata  => w4_m_tdata,
      s_axis_tuser  => w4_m_tuser,
      s_axis_tlast  => w4_m_tlast,

      m_axis_tvalid => m_axis_tvalid,
      m_axis_tready => m_axis_tready,
      m_axis_tdata  => m_axis_tdata,
      m_axis_tuser  => m_axis_tuser,
      m_axis_tlast  => m_axis_tlast
    );

end architecture rtl;
