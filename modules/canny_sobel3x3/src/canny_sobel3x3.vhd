library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library common;

-- 3x3 Sobel gradient magnitude/direction classifier, forked into two
-- independent AXI4-Stream masters from one accepted input beat. See
-- modules/canny_sobel3x3/doc/canny_sobel3x3_req.md and
-- modules/canny_sobel3x3/doc/canny_sobel3x3_proposal.md.
entity canny_sobel3x3 is
  port (
    clk   : in std_logic;
    rst_n : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(71 downto 0);
    s_axis_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_mag_tvalid : out std_logic;
    m_axis_mag_tready : in  std_logic;
    m_axis_mag_tdata  : out std_logic_vector(10 downto 0);
    m_axis_mag_tuser  : out std_logic_vector(1 downto 0);
    m_axis_mag_tlast  : out std_logic;

    m_axis_dir_tvalid : out std_logic;
    m_axis_dir_tready : in  std_logic;
    m_axis_dir_tdata  : out std_logic_vector(1 downto 0);
    m_axis_dir_tuser  : out std_logic_vector(1 downto 0);
    m_axis_dir_tlast  : out std_logic
  );
end entity canny_sobel3x3;

architecture rtl of canny_sobel3x3 is

  constant c_tap_width  : positive := 8;
  constant c_mag_width  : positive := 11;
  constant c_dir_width  : positive := 2;
  constant c_user_width : positive := 2;
  constant c_reg_width  : positive := c_mag_width + c_dir_width + c_user_width + 1;

  -- Signed width with headroom above the provable -1020..1020 range of
  -- Gx/Gy (each term is at most 255*4 = 1020 in magnitude).
  constant c_grad_width : positive := 12;

  -- The 8 taps unpacked from s_axis_tdata that Sobel actually uses
  -- (w_mm, the center tap, is not used by Gx/Gy).
  type window_taps_t is record
    tl, tm, tr, ml, mr, bl, bm, br : unsigned(c_tap_width - 1 downto 0);
  end record;

  -- Same packing as canny_gaussian3x3's input: w_tl=71:64 ... w_br=7:0.
  function unpack_window(tdata : std_logic_vector(71 downto 0)) return window_taps_t is
    variable w : window_taps_t;
  begin
    w.tl := unsigned(tdata(71 downto 64));
    w.tm := unsigned(tdata(63 downto 56));
    w.tr := unsigned(tdata(55 downto 48));
    w.ml := unsigned(tdata(47 downto 40));
    -- tdata(39 downto 32) is w_mm, unused by Sobel.
    w.mr := unsigned(tdata(31 downto 24));
    w.bl := unsigned(tdata(23 downto 16));
    w.bm := unsigned(tdata(15 downto 8));
    w.br := unsigned(tdata(7 downto 0));
    return w;
  end function;

  -- Zero-extend an 8-bit unsigned tap to a c_grad_width-bit signed value.
  function to_grad(u : unsigned(c_tap_width - 1 downto 0)) return signed is
  begin
    return signed(resize(u, c_grad_width));
  end function;

  signal win : window_taps_t;

  signal gx, gy : signed(c_grad_width - 1 downto 0);
  signal ax, ay : unsigned(c_mag_width - 1 downto 0);

  signal mag_sum  : unsigned(c_grad_width - 1 downto 0);
  signal mag_calc : std_logic_vector(c_mag_width - 1 downto 0);
  signal dir_calc : std_logic_vector(c_dir_width - 1 downto 0);

  signal mag_pre : std_logic_vector(c_mag_width - 1 downto 0);
  signal dir_pre : std_logic_vector(c_dir_width - 1 downto 0);

  signal reg_data_in : std_logic_vector(c_reg_width - 1 downto 0);

  -- The one shared register both fork outputs are driven from.
  signal reg_valid : std_logic := '0';
  signal reg_data  : std_logic_vector(c_reg_width - 1 downto 0) := (others => '0');

  signal splitter_input_ready : std_logic;

  signal reg_mag_tdata : std_logic_vector(c_mag_width - 1 downto 0);
  signal reg_dir_tdata : std_logic_vector(c_dir_width - 1 downto 0);
  signal reg_tuser     : std_logic_vector(c_user_width - 1 downto 0);
  signal reg_tlast     : std_logic;

begin

  win <= unpack_window(s_axis_tdata);

  -- Gx = (tr+2*mr+br) - (tl+2*ml+bl); Gy = (bl+2*bm+br) - (tl+2*tm+tr).
  -- Per doc/canny_sobel3x3_req.md's Functional Description.
  gx <= (to_grad(win.tr) + shift_left(to_grad(win.mr), 1) + to_grad(win.br))
    - (to_grad(win.tl) + shift_left(to_grad(win.ml), 1) + to_grad(win.bl));
  gy <= (to_grad(win.bl) + shift_left(to_grad(win.bm), 1) + to_grad(win.br))
    - (to_grad(win.tl) + shift_left(to_grad(win.tm), 1) + to_grad(win.tr));

  ax <= resize(unsigned(abs(gx)), c_mag_width);
  ay <= resize(unsigned(abs(gy)), c_mag_width);

  -- Max ax+ay = 1020+1020 = 2040, fits unsigned 11-bit (max 2047) exactly.
  mag_sum  <= resize(ax, c_grad_width) + resize(ay, c_grad_width);
  mag_calc <= std_logic_vector(resize(mag_sum, c_mag_width));

  -- "00" is checked first, so ax=ay=0 resolves to "00" (explicit tie-break
  -- rule in the requirement). gx(gx'high)=gy(gy'high) is equivalent to
  -- (gx >= 0) = (gy >= 0) since both are signed sign bits.
  dir_calc <=
    "00" when ay <= shift_right(ax, 1) else
    "10" when ax <= shift_right(ay, 1) else
    "01" when gx(gx'high) = gy(gy'high) else
    "11";

  -- Border (s_axis_tuser(1)='1') forces mag=0, dir="00" after the raw calc;
  -- tuser/tlast always pass through unchanged (handled below in reg_data_in).
  mag_pre <= (others => '0') when s_axis_tuser(1) = '1' else mag_calc;
  dir_pre <= "00" when s_axis_tuser(1) = '1' else dir_calc;

  reg_data_in <= mag_pre & dir_pre & s_axis_tuser & s_axis_tlast;

  -- Hand-written one-cycle elastic register (rst_n-clearable, unlike
  -- common.handshake_pipeline which has no reset port -- see proposal
  -- doc). Accepts a new input beat whenever the register is empty (not
  -- reg_valid) or is about to be freed this cycle (splitter_input_ready,
  -- i.e. both forks accepted the current contents).
  s_axis_tready <= splitter_input_ready or (not reg_valid);

  register_proc : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        reg_valid <= '0';
      elsif s_axis_tvalid = '1' and s_axis_tready = '1' then
        reg_data  <= reg_data_in;
        reg_valid <= '1';
      elsif splitter_input_ready = '1' then
        reg_valid <= '0';
      end if;
    end if;
  end process;

  reg_mag_tdata <= reg_data(c_reg_width - 1 downto c_dir_width + c_user_width + 1);
  reg_dir_tdata <= reg_data(c_dir_width + c_user_width downto c_user_width + 1);
  reg_tuser     <= reg_data(c_user_width downto 1);
  reg_tlast     <= reg_data(0);

  ------------------------------------------------------------------------------
  handshake_splitter_inst : entity common.handshake_splitter
    generic map (
      num_interfaces => 2
    )
    port map (
      clk => clk,
      --
      input_ready => splitter_input_ready,
      input_valid => reg_valid,
      --
      output_ready(0) => m_axis_mag_tready,
      output_ready(1) => m_axis_dir_tready,
      output_valid(0) => m_axis_mag_tvalid,
      output_valid(1) => m_axis_dir_tvalid
    );

  m_axis_mag_tdata <= reg_mag_tdata;
  m_axis_mag_tuser <= reg_tuser;
  m_axis_mag_tlast <= reg_tlast;

  m_axis_dir_tdata <= reg_dir_tdata;
  m_axis_dir_tuser <= reg_tuser;
  m_axis_dir_tlast <= reg_tlast;

end architecture rtl;
