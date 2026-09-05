library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library common;

-- Non-maximum suppression along the gradient direction sector. See
-- modules/canny/doc/canny_nms_req.md and
-- modules/canny/doc/canny_nms_proposal.md.
entity canny_nms is
  port (
    clk   : in std_logic;
    reset : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(100 downto 0);
    s_axis_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(10 downto 0);
    m_axis_tuser  : out std_logic_vector(1 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity canny_nms;

architecture a of canny_nms is

  constant mag_width : positive := 11;

  -- Packed handshake_pipeline payload: nms_result & s_axis_tuser (mirrors
  -- canny_window3x3's pack_lane convention -- data bits, then passenger
  -- bits). s_axis_tlast rides handshake_pipeline's own dedicated
  -- input_last/output_last port instead of being packed in here.
  constant pipeline_width : positive := mag_width + 2;

  subtype magnitude_t is std_logic_vector(mag_width - 1 downto 0);

  signal direction_in : std_logic_vector(1 downto 0);

  signal w_tl, w_tm, w_tr : magnitude_t;
  signal w_ml, w_mm, w_mr : magnitude_t;
  signal w_bl, w_bm, w_br : magnitude_t;

  signal nms_result : magnitude_t;

  signal pipeline_input_data   : std_logic_vector(pipeline_width - 1 downto 0);
  signal pipeline_output_data  : std_logic_vector(pipeline_width - 1 downto 0);
  signal pipeline_output_valid : std_logic;

begin

  direction_in <= s_axis_tdata(1 downto 0);

  -- 3x3 magnitude window taps, unpacked from s_axis_tdata(100 downto 2)
  -- using the same 9x11 row-major MSB-to-LSB packing canny_window3x3
  -- produces for its own m_axis_tdata.
  w_tl <= s_axis_tdata(100 downto 90);
  w_tm <= s_axis_tdata(89 downto 79);
  w_tr <= s_axis_tdata(78 downto 68);
  w_ml <= s_axis_tdata(67 downto 57);
  w_mm <= s_axis_tdata(56 downto 46);
  w_mr <= s_axis_tdata(45 downto 35);
  w_bl <= s_axis_tdata(34 downto 24);
  w_bm <= s_axis_tdata(23 downto 13);
  w_br <= s_axis_tdata(12 downto 2);

  ------------------------------------------------------------------------------
  -- Combinational NMS compare: keep w_mm only if it is a local maximum
  -- along the gradient direction sector selected by direction_in; else
  -- suppress to 0. Forced to 0 unconditionally at the border.
  compare : process(all)
    variable neighbor_a, neighbor_b : magnitude_t;
  begin
    case direction_in is
      when "00" =>
        neighbor_a := w_ml;
        neighbor_b := w_mr;
      when "01" =>
        neighbor_a := w_tr;
        neighbor_b := w_bl;
      when "10" =>
        neighbor_a := w_tm;
        neighbor_b := w_bm;
      when others =>
        neighbor_a := w_tl;
        neighbor_b := w_br;
    end case;

    if s_axis_tuser(1) = '1' then
      nms_result <= (others => '0');
    elsif unsigned(w_mm) >= unsigned(neighbor_a) and unsigned(w_mm) >= unsigned(neighbor_b) then
      nms_result <= w_mm;
    else
      nms_result <= (others => '0');
    end if;
  end process;

  pipeline_input_data <= nms_result & s_axis_tuser;

  ------------------------------------------------------------------------------
  handshake_pipeline_inst : entity common.handshake_pipeline
    generic map (
      data_width => pipeline_width
    )
    port map (
      clk => clk,
      --
      input_ready => s_axis_tready,
      input_valid => s_axis_tvalid,
      input_last  => s_axis_tlast,
      input_data  => pipeline_input_data,
      --
      output_ready => m_axis_tready,
      output_valid => pipeline_output_valid,
      output_last  => m_axis_tlast,
      output_data  => pipeline_output_data
    );

  -- common.handshake_pipeline has no reset port of its own (verified via
  -- vhdl-rag-mcp against modules/common/src/handshake_pipeline.vhd); gate
  -- the externally visible tvalid with reset so the "output register"
  -- reads as cleared during reset, per doc/canny_nms_req.md's clock/reset
  -- section. See doc/canny_nms_proposal.md "Clock/reset behavior".
  m_axis_tvalid <= pipeline_output_valid and not reset;

  m_axis_tdata <= pipeline_output_data(pipeline_width - 1 downto 2);
  m_axis_tuser <= pipeline_output_data(1 downto 0);

end architecture a;
