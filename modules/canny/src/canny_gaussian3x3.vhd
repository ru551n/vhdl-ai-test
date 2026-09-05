library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library common;

-- 3x3 approximate Gaussian smoothing of the raw pixel window, as a full
-- AXI4-Stream elastic stage. See
-- modules/canny/doc/canny_gaussian3x3_req.md and
-- modules/canny/doc/canny_gaussian3x3_proposal.md.
--
-- Resetless: common.handshake_pipeline (the only piece of sequential state
-- this module owns) has no reset port at all, and no other state needs
-- restoring. See doc/canny_gaussian3x3.md "Clocking and reset".
entity canny_gaussian3x3 is
  port (
    clk : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(71 downto 0);
    s_axis_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(7 downto 0);
    m_axis_tuser  : out std_logic_vector(1 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity canny_gaussian3x3;

architecture a of canny_gaussian3x3 is

  constant tap_width : positive := 8;

  -- input_data width for the single handshake_pipeline instance: computed
  -- pixel result (8) + s_axis_tuser (2) + s_axis_tlast (1). tuser/tlast
  -- ride through as extra data-bit lanes packed alongside the computed
  -- pixel, not handshake_pipeline's own separate input_last/output_last
  -- ports -- see doc/canny_gaussian3x3.md "Implementation notes".
  constant pipeline_data_width : positive := tap_width + 2 + 1;

  -- Combinationally computes the 3x3 approximate-Gaussian weighted sum
  -- per canny_gaussian3x3_req.md's "Functional Description":
  -- (tl + 2*tm + tr + 2*ml + 4*mm + 2*mr + bl + 2*bm + br) / 16.
  function gaussian_weighted_sum(
    tl, tm, tr, ml, mm, mr, bl, bm, br : unsigned(tap_width - 1 downto 0)
  ) return unsigned is
    -- Worst case 16*255=4080 fits in 12 bits.
    constant sum_width : positive := tap_width + 4;
    variable weighted_sum : unsigned(sum_width - 1 downto 0);
  begin
    weighted_sum := resize(tl, sum_width)
                  + resize(tm, sum_width) + resize(tm, sum_width)
                  + resize(tr, sum_width)
                  + resize(ml, sum_width) + resize(ml, sum_width)
                  + resize(mm, sum_width) + resize(mm, sum_width)
                  + resize(mm, sum_width) + resize(mm, sum_width)
                  + resize(mr, sum_width) + resize(mr, sum_width)
                  + resize(bl, sum_width)
                  + resize(bm, sum_width) + resize(bm, sum_width)
                  + resize(br, sum_width);

    -- Exact power-of-two divide by 16: truncating right-shift, no rounding.
    return resize(shift_right(weighted_sum, 4), tap_width);
  end function;

  signal w_tl, w_tm, w_tr, w_ml, w_mm, w_mr, w_bl, w_bm, w_br
    : unsigned(tap_width - 1 downto 0);

  signal gaussian_result : unsigned(tap_width - 1 downto 0);
  signal masked_result : std_logic_vector(tap_width - 1 downto 0);

  signal pipeline_input_data, pipeline_output_data
    : std_logic_vector(pipeline_data_width - 1 downto 0);

begin

  -- Unpack the 9 taps from s_axis_tdata, row-major MSB-to-LSB per the
  -- requirement's port table.
  w_tl <= unsigned(s_axis_tdata(71 downto 64));
  w_tm <= unsigned(s_axis_tdata(63 downto 56));
  w_tr <= unsigned(s_axis_tdata(55 downto 48));
  w_ml <= unsigned(s_axis_tdata(47 downto 40));
  w_mm <= unsigned(s_axis_tdata(39 downto 32));
  w_mr <= unsigned(s_axis_tdata(31 downto 24));
  w_bl <= unsigned(s_axis_tdata(23 downto 16));
  w_bm <= unsigned(s_axis_tdata(15 downto 8));
  w_br <= unsigned(s_axis_tdata(7 downto 0));

  gaussian_result <= gaussian_weighted_sum(
    tl => w_tl, tm => w_tm, tr => w_tr,
    ml => w_ml, mm => w_mm, mr => w_mr,
    bl => w_bl, bm => w_bm, br => w_br
  );

  -- Force masked_result to x"00" at the border (s_axis_tuser(1) = '1'),
  -- else pass through the computed weighted sum, per requirement.
  masked_result <= (others => '0') when s_axis_tuser(1) = '1'
               else std_logic_vector(gaussian_result);

  pipeline_input_data <= masked_result & s_axis_tuser & s_axis_tlast;

  m_axis_tdata <= pipeline_output_data(pipeline_data_width - 1 downto 3);
  m_axis_tuser <= pipeline_output_data(2 downto 1);
  m_axis_tlast <= pipeline_output_data(0);

  ------------------------------------------------------------------------------
  handshake_pipeline_inst : entity common.handshake_pipeline
    generic map (
      data_width => pipeline_data_width
    )
    port map (
      clk => clk,
      --
      input_ready => s_axis_tready,
      input_valid => s_axis_tvalid,
      input_data => pipeline_input_data,
      --
      output_ready => m_axis_tready,
      output_valid => m_axis_tvalid,
      output_data => pipeline_output_data
    );

end architecture a;
